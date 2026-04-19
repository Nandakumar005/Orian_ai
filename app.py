import json
import os
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def get_game_id() -> str:
    if "game_id" not in session:
        session["game_id"] = str(uuid.uuid4())
    return session["game_id"]


def state_path(game_id: str) -> Path:
    return DATA_DIR / f"{game_id}.json"


def default_state():
    return {
        "scene": {
            "title": "The First Door",
            "text": "You wake inside a stone chamber. A cold wind comes from the dark corridor ahead.",
            "location": "Stone Chamber",
        },
        "character": {
            "name": "Old Gatekeeper",
            "role": "mysterious guide",
            "mood": "watchful",
            "note": "He knows more than he says.",
        },
        "memory": {
            "facts": ["The player woke up in a dungeon.", "There is a dark corridor ahead."],
            "inventory": ["Rusty torch"],
            "quests": ["Find the exit"],
            "flags": [],
        },
        "story_summary": "The player has just entered a dungeon and met an old gatekeeper.",
        "choices": [
            {"id": "explore", "label": "Explore the corridor"},
            {"id": "talk", "label": "Talk to the gatekeeper"},
            {"id": "inspect", "label": "Inspect the room"},
        ],
        "history": [],
    }


def load_state(game_id: str):
    path = state_path(game_id)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default_state()


def save_state(game_id: str, state: dict):
    path = state_path(game_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def merge_list_unique(target_list, items):
    for item in items or []:
        if item not in target_list:
            target_list.append(item)


def remove_from_list(target_list, items):
    for item in items or []:
        if item in target_list:
            target_list.remove(item)


def merge_memory(state: dict, memory_update: dict):
    mem = state["memory"]
    merge_list_unique(mem["facts"], memory_update.get("facts_add"))
    merge_list_unique(mem["inventory"], memory_update.get("inventory_add"))
    merge_list_unique(mem["quests"], memory_update.get("quests_add"))
    merge_list_unique(mem["flags"], memory_update.get("flags_add"))

    remove_from_list(mem["inventory"], memory_update.get("inventory_remove"))
    remove_from_list(mem["quests"], memory_update.get("quests_remove"))
    remove_from_list(mem["flags"], memory_update.get("flags_remove"))


def build_messages(state: dict, player_action: str):
    system_prompt = f"""
You are the dungeon master for a browser-based AI dungeon game.

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{{
  "scene_title": "string",
  "scene_text": "string",
  "character": {{
    "name": "string",
    "role": "string",
    "mood": "string",
    "note": "string"
  }},
  "choices": [
    {{"id": "string", "label": "string"}},
    {{"id": "string", "label": "string"}},
    {{"id": "string", "label": "string"}}
  ],
  "memory_update": {{
    "facts_add": ["string"],
    "inventory_add": ["string"],
    "inventory_remove": ["string"],
    "quests_add": ["string"],
    "quests_remove": ["string"],
    "flags_add": ["string"],
    "flags_remove": ["string"]
  }}
}}

Rules:
- Keep the adventure consistent with the persistent memory.
- Make the scene vivid, interactive, and short enough for a web card.
- Always provide exactly 3 choices.
- Choices must be actions the player can actually take.
- Do not forget important story facts.
- Current persistent memory:
{json.dumps(state["memory"], ensure_ascii=False)}

- Current story summary:
{state["story_summary"]}

- Recent history:
{json.dumps(state["history"][-8:], ensure_ascii=False)}
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": f"Player action: {player_action}"})
    return messages


def call_ollama(state: dict, player_action: str):
    payload = {
        "model": MODEL,
        "messages": build_messages(state, player_action),
        "format": "json",
        "stream": False,
        "keep_alive": "5m",
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    return json.loads(content)


def fallback_turn(state: dict, player_action: str):
    scene_num = len(state["history"]) // 2 + 1
    return {
        "scene_title": f"Deepening Path {scene_num}",
        "scene_text": f"You chose to {player_action.lower()}. The dungeon shifts around you, and a new passage opens ahead.",
        "character": state["character"],
        "choices": [
            {"id": "left", "label": "Take the left passage"},
            {"id": "right", "label": "Take the right passage"},
            {"id": "wait", "label": "Wait and listen"},
        ],
        "memory_update": {
            "facts_add": [f"The player chose to {player_action.lower()}."],
            "inventory_add": [],
            "inventory_remove": [],
            "quests_add": [],
            "quests_remove": [],
            "flags_add": [],
            "flags_remove": [],
        },
    }


def run_turn(state: dict, player_action: str):
    try:
        result = call_ollama(state, player_action)
        if not isinstance(result, dict):
            raise ValueError("Model did not return a JSON object.")
    except Exception:
        result = fallback_turn(state, player_action)

    state["scene"] = {
        "title": result.get("scene_title", "Unknown Chamber"),
        "text": result.get("scene_text", ""),
        "location": result.get("location", state["scene"].get("location", "")),
    }
    state["character"] = result.get("character", state["character"])
    state["choices"] = result.get("choices", state["choices"])

    memory_update = result.get("memory_update", {})
    merge_memory(state, memory_update)

    state["story_summary"] = (
        state["story_summary"] + " " + state["scene"]["text"]
    ).strip()[-1500:]

    state["history"].append({"role": "user", "content": player_action})
    state["history"].append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
    state["history"] = state["history"][-12:]

    return state


def ensure_started(state: dict):
    if not state["history"]:
        state = run_turn(state, "Start the adventure")
    return state


@app.get("/")
def index():
    game_id = get_game_id()
    state = load_state(game_id)
    state = ensure_started(state)
    save_state(game_id, state)
    return render_template("index.html", state=state)


@app.get("/api/state")
def api_state():
    game_id = get_game_id()
    state = load_state(game_id)
    state = ensure_started(state)
    save_state(game_id, state)
    return jsonify(state)


@app.post("/api/action")
def api_action():
    game_id = get_game_id()
    state = load_state(game_id)

    data = request.get_json(force=True)
    action = (data.get("action") or "").strip()
    if not action:
        return jsonify({"error": "Action cannot be empty"}), 400

    state = run_turn(state, action)
    save_state(game_id, state)
    return jsonify(state)


@app.post("/api/new")
def api_new():
    game_id = get_game_id()
    state = default_state()
    state = ensure_started(state)
    save_state(game_id, state)
    return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True)
import json
import os
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def get_game_id() -> str:
    if "game_id" not in session:
        session["game_id"] = str(uuid.uuid4())
    return session["game_id"]


def state_path(game_id: str) -> Path:
    return DATA_DIR / f"{game_id}.json"


def default_state():
    return {
        "scene": {
            "title": "The First Door",
            "text": "You wake inside a stone chamber. A cold wind comes from the dark corridor ahead.",
            "location": "Stone Chamber",
        },
        "character": {
            "name": "Old Gatekeeper",
            "role": "mysterious guide",
            "mood": "watchful",
            "note": "He knows more than he says.",
        },
        "memory": {
            "facts": ["The player woke up in a dungeon.", "There is a dark corridor ahead."],
            "inventory": ["Rusty torch"],
            "quests": ["Find the exit"],
            "flags": [],
        },
        "story_summary": "The player has just entered a dungeon and met an old gatekeeper.",
        "choices": [
            {"id": "explore", "label": "Explore the corridor"},
            {"id": "talk", "label": "Talk to the gatekeeper"},
            {"id": "inspect", "label": "Inspect the room"},
        ],
        "history": [],
    }


def load_state(game_id: str):
    path = state_path(game_id)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default_state()


def save_state(game_id: str, state: dict):
    path = state_path(game_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def merge_list_unique(target_list, items):
    for item in items or []:
        if item not in target_list:
            target_list.append(item)


def remove_from_list(target_list, items):
    for item in items or []:
        if item in target_list:
            target_list.remove(item)


def merge_memory(state: dict, memory_update: dict):
    mem = state["memory"]
    merge_list_unique(mem["facts"], memory_update.get("facts_add"))
    merge_list_unique(mem["inventory"], memory_update.get("inventory_add"))
    merge_list_unique(mem["quests"], memory_update.get("quests_add"))
    merge_list_unique(mem["flags"], memory_update.get("flags_add"))

    remove_from_list(mem["inventory"], memory_update.get("inventory_remove"))
    remove_from_list(mem["quests"], memory_update.get("quests_remove"))
    remove_from_list(mem["flags"], memory_update.get("flags_remove"))


def build_messages(state: dict, player_action: str):
    system_prompt = f"""
You are the dungeon master for a browser-based AI dungeon game.

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{{
  "scene_title": "string",
  "scene_text": "string",
  "character": {{
    "name": "string",
    "role": "string",
    "mood": "string",
    "note": "string"
  }},
  "choices": [
    {{"id": "string", "label": "string"}},
    {{"id": "string", "label": "string"}},
    {{"id": "string", "label": "string"}}
  ],
  "memory_update": {{
    "facts_add": ["string"],
    "inventory_add": ["string"],
    "inventory_remove": ["string"],
    "quests_add": ["string"],
    "quests_remove": ["string"],
    "flags_add": ["string"],
    "flags_remove": ["string"]
  }}
}}

Rules:
- Keep the adventure consistent with the persistent memory.
- Make the scene vivid, interactive, and short enough for a web card.
- Always provide exactly 3 choices.
- Choices must be actions the player can actually take.
- Do not forget important story facts.
- Current persistent memory:
{json.dumps(state["memory"], ensure_ascii=False)}

- Current story summary:
{state["story_summary"]}

- Recent history:
{json.dumps(state["history"][-8:], ensure_ascii=False)}
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": f"Player action: {player_action}"})
    return messages


def call_ollama(state: dict, player_action: str):
    payload = {
        "model": MODEL,
        "messages": build_messages(state, player_action),
        "format": "json",
        "stream": False,
        "keep_alive": "5m",
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    return json.loads(content)


def fallback_turn(state: dict, player_action: str):
    scene_num = len(state["history"]) // 2 + 1
    return {
        "scene_title": f"Deepening Path {scene_num}",
        "scene_text": f"You chose to {player_action.lower()}. The dungeon shifts around you, and a new passage opens ahead.",
        "character": state["character"],
        "choices": [
            {"id": "left", "label": "Take the left passage"},
            {"id": "right", "label": "Take the right passage"},
            {"id": "wait", "label": "Wait and listen"},
        ],
        "memory_update": {
            "facts_add": [f"The player chose to {player_action.lower()}."],
            "inventory_add": [],
            "inventory_remove": [],
            "quests_add": [],
            "quests_remove": [],
            "flags_add": [],
            "flags_remove": [],
        },
    }


def run_turn(state: dict, player_action: str):
    try:
        result = call_ollama(state, player_action)
        if not isinstance(result, dict):
            raise ValueError("Model did not return a JSON object.")
    except Exception:
        result = fallback_turn(state, player_action)

    state["scene"] = {
        "title": result.get("scene_title", "Unknown Chamber"),
        "text": result.get("scene_text", ""),
        "location": result.get("location", state["scene"].get("location", "")),
    }
    state["character"] = result.get("character", state["character"])
    state["choices"] = result.get("choices", state["choices"])

    memory_update = result.get("memory_update", {})
    merge_memory(state, memory_update)

    state["story_summary"] = (
        state["story_summary"] + " " + state["scene"]["text"]
    ).strip()[-1500:]

    state["history"].append({"role": "user", "content": player_action})
    state["history"].append({"role": "assistant", "content": json.dumps(result, ensure_ascii=False)})
    state["history"] = state["history"][-12:]

    return state


def ensure_started(state: dict):
    if not state["history"]:
        state = run_turn(state, "Start the adventure")
    return state


@app.get("/")
def index():
    game_id = get_game_id()
    state = load_state(game_id)
    state = ensure_started(state)
    save_state(game_id, state)
    return render_template("index.html", state=state)


@app.get("/api/state")
def api_state():
    game_id = get_game_id()
    state = load_state(game_id)
    state = ensure_started(state)
    save_state(game_id, state)
    return jsonify(state)


@app.post("/api/action")
def api_action():
    game_id = get_game_id()
    state = load_state(game_id)

    data = request.get_json(force=True)
    action = (data.get("action") or "").strip()
    if not action:
        return jsonify({"error": "Action cannot be empty"}), 400

    state = run_turn(state, action)
    save_state(game_id, state)
    return jsonify(state)


@app.post("/api/new")
def api_new():
    game_id = get_game_id()
    state = default_state()
    state = ensure_started(state)
    save_state(game_id, state)
    return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True)
"""
Tools module for embedded assistant
All tool names are max 4 characters for compact DSL
"""

from typing import Dict, Any, Optional
import json
import math
import random

# Family key-value store (in-memory for demo, would be persistent in real system)
family_store: Dict[str, Any] = {
    "oliwia": {"birthday": "2015-07-29"},
    "dad": {"birthday": "1981-05-15"},
    "mom": {"birthday": "1984-08-12"},
    "light": {"state": "off"},
    "garage": {"door": "closed"},
}

def math_tool(expression: str) -> str:
    """Math tool - evaluate mathematical expressions (max 4 chars: math)"""
    try:
        # Safe evaluation - only allow basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "E:Invalid characters"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"E:{str(e)}"

def wthr_tool(location: str = "", date: str = "") -> str:
    """Weather tool - get weather info (max 4 chars: wthr)"""
    from datetime import datetime, timedelta
    
    # If date is a relative date (tomorrow, yesterday, etc.), resolve it first
    if date and date.lower() in ["today", "tomorrow", "yesterday"] or date.startswith("+") or date.startswith("-"):
        date = date_tool(date)
    
    # Default to today if no date provided
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Simulated weather data - can vary by date
    location = location.lower() if location else ""
    
    # Basic weather data
    base_weather = {
        "": "72F, sunny",
        "home": "72F, sunny",
        "outside": "68F, partly cloudy",
        "kitchen": "70F, warm",
    }
    
    # Add date-specific variations (simulated)
    weather = base_weather.get(location, "72F, sunny")
    
    # Format: weather for date
    if date:
        return f"{weather} on {date}"
    return weather

def fam_tool(action: str, key: str = "", value: str = "") -> str:
    """Family key-value store tool (max 4 chars: fam)"""
    global family_store
    
    action = action.lower()
    
    if action == "get":
        if key in family_store:
            return json.dumps(family_store[key])
        return "E:Key not found"
    
    elif action == "set":
        if not key:
            return "E:Key required"
        try:
            # Try to parse as JSON, otherwise store as string
            try:
                parsed_value = json.loads(value)
                family_store[key] = parsed_value
            except:
                family_store[key] = value
            return "OK"
        except Exception as e:
            return f"E:{str(e)}"
    
    elif action == "list":
        return json.dumps(list(family_store.keys()))
    
    elif action == "where":
        # Special: where is person
        if key in family_store and "location" in family_store[key]:
            return family_store[key]["location"]
        return "E:Person not found"
    
    elif action == "age":
        # Special: how old is person
        if key in family_store and "age" in family_store[key]:
            return str(family_store[key]["age"])
        return "E:Person not found"
    
    return "E:Invalid action"

def lght_tool(action: str = "") -> str:
    """Light control tool (max 4 chars: lght)"""
    global family_store
    
    if not action:
        # Get state
        return family_store.get("light", {}).get("state", "off")
    
    action = action.lower()
    if action in ["on", "off"]:
        if "light" not in family_store:
            family_store["light"] = {}
        family_store["light"]["state"] = action
        return f"Light turned {action}"
    
    return "E:Invalid action (use 'on' or 'off')"

def gdge_tool(action: str = "") -> str:
    """Garage door tool (max 4 chars: gdge)"""
    global family_store
    
    if not action:
        # Get state
        state = family_store.get("garage", {}).get("door", "closed")
        return f"Garage door is {state}"
    
    action = action.lower()
    if action in ["open", "close"]:
        if "garage" not in family_store:
            family_store["garage"] = {}
        family_store["garage"]["door"] = action + "d"
        return f"Garage door {action}ed"
    
    return "E:Invalid action (use 'open' or 'close')"

def move_tool(location: str) -> str:
    """Movement command tool (max 4 chars: move)"""
    valid_locations = ["kitchen", "bedroom", "living room", "office", "bathroom", "garage"]
    location = location.lower()
    
    if location in valid_locations:
        return f"Moving to {location}"
    return f"E:Invalid location. Valid: {', '.join(valid_locations)}"

def time_tool() -> str:
    """Time tool - get current time (max 4 chars: time)"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M")

def date_tool(relative: str = "") -> str:
    """Date tool - get date (max 4 chars: date). Supports relative dates like 'today', 'tomorrow', 'yesterday'"""
    from datetime import datetime, timedelta
    
    today = datetime.now()
    relative = relative.lower().strip() if relative else "today"
    
    if relative == "today" or relative == "":
        return today.strftime("%Y-%m-%d")
    elif relative == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif relative == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif relative.startswith("+") or relative.startswith("-"):
        # Handle +/-N days format
        try:
            days = int(relative)
            return (today + timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            return f"E:Invalid date format: {relative}"
    else:
        # Try to parse as YYYY-MM-DD format
        try:
            # Validate date format
            datetime.strptime(relative, "%Y-%m-%d")
            return relative
        except ValueError:
            return f"E:Invalid date format: {relative}"

def calc_tool(expression: str) -> str:
    """Calculator tool - advanced math (max 4 chars: calc)"""
    try:
        # More advanced math functions
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"E:{str(e)}"

def remd_tool(text: str) -> str:
    """Reminder tool - set reminder (max 4 chars: remd)"""
    # In real system, would store reminders
    return f"Reminder set: {text}"

def alarm_tool(action: str, time: str = "") -> str:
    """Alarm tool (max 4 chars: alarm)"""
    if action == "set" and time:
        return f"Alarm set for {time}"
    elif action == "cancel":
        return "Alarm cancelled"
    return "E:Invalid action"

# Tool registry - maps tool names to functions
TOOLS: Dict[str, callable] = {
    "math": math_tool,
    "wthr": wthr_tool,
    "fam": fam_tool,
    "lght": lght_tool,
    "gdge": gdge_tool,
    "move": move_tool,
    "time": time_tool,
    "date": date_tool,
    "calc": calc_tool,
    "remd": remd_tool,
    "alrm": alarm_tool,
}

def execute_tool(tool_name: str, args: str = "") -> str:
    """Execute a tool by name"""
    if tool_name not in TOOLS:
        return f"E:Tool '{tool_name}' not found"
    
    try:
        tool_func = TOOLS[tool_name]
        
        # Parse args based on tool
        if args:
            # Some tools take multiple args separated by comma
            if tool_name == "fam":
                parts = args.split(",", 2)
                if len(parts) == 1:
                    return tool_func(parts[0])
                elif len(parts) == 2:
                    return tool_func(parts[0], parts[1])
                else:
                    return tool_func(parts[0], parts[1], parts[2])
            elif tool_name == "wthr":
                # Weather tool: can take location,date or just location or just date
                # If args contains a comma, it's location,date format
                if "," in args:
                    parts = args.split(",", 1)
                    return tool_func(parts[0], parts[1])
                else:
                    # Single arg - could be location or date
                    # If it looks like a date (YYYY-MM-DD or relative like "tomorrow"), treat as date
                    if args and (args.startswith("20") or args in ["today", "tomorrow", "yesterday"] or args.startswith("+") or args.startswith("-")):
                        return tool_func("", args)  # date only
                    else:
                        return tool_func(args, "")  # location only
            elif tool_name in ["lght", "gdge", "alrm"]:
                # Single arg tools
                return tool_func(args)
            elif tool_name == "date":
                # Date tool with relative date
                return tool_func(args)
            else:
                # Default: pass args as single string
                return tool_func(args)
        else:
            # No args
            if tool_name in ["time", "date"]:
                return tool_func()
            elif tool_name in ["lght", "gdge"]:
                return tool_func("")
            elif tool_name == "wthr":
                return tool_func("", "")
            else:
                return tool_func("")
    
    except Exception as e:
        return f"E:{str(e)}"

if __name__ == "__main__":
    # Test tools
    print("Testing tools:")
    print(f"math(2+2) = {math_tool('2+2')}")
    print(f"wthr() = {wthr_tool()}")
    print(f"fam(where, oliwia) = {fam_tool('where', 'oliwia')}")
    print(f"fam(age, oliwia) = {fam_tool('age', 'oliwia')}")
    print(f"lght(on) = {lght_tool('on')}")
    print(f"gdge() = {gdge_tool()}")
    print(f"move(kitchen) = {move_tool('kitchen')}")
    print(f"time() = {time_tool()}")


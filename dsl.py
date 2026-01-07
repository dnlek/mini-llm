"""
DSL Parser for Embedded Assistant System
Compact format for embedded systems with minimal overhead
"""

import re
from typing import List, Dict, Optional, Tuple, Union

# DSL Format:
# S:text - Simple response
# T:tool,args - Tool call
# C:cmd,args - Command
# S:text;C:cmd,args - Response + Command
# R:result - Tool result (used in recursive calls)
# CL - Cloud route
# E:error - Error message

class DSLEncoder:
    """Encode natural language to DSL format"""
    
    @staticmethod
    def encode_response(text: str) -> str:
        """Encode a simple response"""
        return f"S:{text}"
    
    @staticmethod
    def encode_tool(tool_name: str, args: str = "") -> str:
        """Encode a tool call"""
        if args:
            return f"T:{tool_name},{args}"
        return f"T:{tool_name}"
    
    @staticmethod
    def encode_command(cmd: str, args: str = "") -> str:
        """Encode a command"""
        if args:
            return f"C:{cmd},{args}"
        return f"C:{cmd}"
    
    @staticmethod
    def encode_response_command(response: str, cmd: str, args: str = "") -> str:
        """Encode response + command"""
        cmd_part = DSLEncoder.encode_command(cmd, args)
        return f"S:{response};{cmd_part}"
    
    @staticmethod
    def encode_cloud() -> str:
        """Encode cloud route"""
        return "CL"
    
    @staticmethod
    def encode_error(error: str) -> str:
        """Encode error message"""
        return f"E:{error}"
    
    @staticmethod
    def encode_tool_chain(tools: List[Tuple[str, str]]) -> str:
        """Encode a chain of tool calls. tools is a list of (tool_name, args) tuples"""
        tool_parts = []
        for tool_name, args in tools:
            if args:
                tool_parts.append(f"T:{tool_name},{args}")
            else:
                tool_parts.append(f"T:{tool_name}")
        return ";".join(tool_parts)


class DSLDecoder:
    """Decode DSL format to structured data"""
    
    @staticmethod
    def decode(dsl_string: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Decode DSL string to structured format
        Returns: {
            'type': 'response' | 'tool' | 'command' | 'response_command' | 'tool_chain' | 'cloud' | 'error',
            'content': str or list of dicts
        }
        """
        dsl_string = dsl_string.strip()
        
        # Cloud route
        if dsl_string == "CL":
            return {'type': 'cloud', 'content': ''}
        
        # Multiple parts (could be response + command, or tool chain)
        if ';' in dsl_string:
            parts = dsl_string.split(';')
            decoded_parts = []
            has_tool = False
            has_response = False
            has_command = False
            
            for part in parts:
                decoded = DSLDecoder._decode_single(part.strip())
                if decoded:
                    decoded_parts.append(decoded)
                    if decoded.get('type') == 'tool':
                        has_tool = True
                    elif decoded.get('type') == 'response':
                        has_response = True
                    elif decoded.get('type') == 'command':
                        has_command = True
            
            # Determine type: tool_chain if all are tools, otherwise response_command
            if has_tool and not has_response and not has_command:
                return {'type': 'tool_chain', 'content': decoded_parts}
            else:
                return {'type': 'response_command', 'content': decoded_parts}
        
        # Single part
        return DSLDecoder._decode_single(dsl_string)
    
    @staticmethod
    def _decode_single(dsl_string: str) -> Optional[Dict[str, str]]:
        """Decode a single DSL component"""
        if not dsl_string:
            return None
        
        # Match pattern: TYPE:content or TYPE:tool,args
        match = re.match(r'^([A-Z]+):(.+)$', dsl_string)
        if not match:
            return None
        
        dsl_type = match.group(1)
        content = match.group(2)
        
        if dsl_type == 'S':
            return {'type': 'response', 'text': content}
        elif dsl_type == 'T':
            # Tool: T:tool,args or T:tool
            if ',' in content:
                tool, args = content.split(',', 1)
                return {'type': 'tool', 'tool': tool, 'args': args}
            else:
                return {'type': 'tool', 'tool': content, 'args': ''}
        elif dsl_type == 'C':
            # Command: C:cmd,args or C:cmd
            if ',' in content:
                cmd, args = content.split(',', 1)
                return {'type': 'command', 'command': cmd, 'args': args}
            else:
                return {'type': 'command', 'command': content, 'args': ''}
        elif dsl_type == 'E':
            return {'type': 'error', 'error': content}
        elif dsl_type == 'CL':
            return {'type': 'cloud', 'content': ''}
        
        return None
    
    @staticmethod
    def is_tool(dsl_string: str) -> bool:
        """Check if DSL string is a tool call"""
        decoded = DSLDecoder.decode(dsl_string)
        if decoded['type'] == 'tool':
            return True
        if decoded['type'] == 'tool_chain':
            return True
        if decoded['type'] == 'response_command':
            return any(item.get('type') == 'tool' for item in decoded['content'])
        return False
    
    @staticmethod
    def extract_tool_chain(dsl_string: str) -> Optional[List[Tuple[str, str]]]:
        """Extract tool chain from DSL string. Returns list of (tool_name, args) tuples"""
        decoded = DSLDecoder.decode(dsl_string)
        
        if decoded['type'] == 'tool_chain':
            return [(item['tool'], item.get('args', '')) for item in decoded['content']]
        
        if decoded['type'] == 'tool':
            return [(decoded['tool'], decoded.get('args', ''))]
        
        return None
    
    @staticmethod
    def extract_tool(dsl_string: str) -> Optional[Tuple[str, str]]:
        """Extract tool name and args from DSL string"""
        decoded = DSLDecoder.decode(dsl_string)
        
        if decoded['type'] == 'tool':
            return (decoded['tool'], decoded.get('args', ''))
        
        if decoded['type'] == 'response_command':
            for item in decoded['content']:
                if item.get('type') == 'tool':
                    return (item['tool'], item.get('args', ''))
        
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test encoding
    print("Encoding tests:")
    print(DSLEncoder.encode_response("Hello"))
    print(DSLEncoder.encode_tool("math", "2+2"))
    print(DSLEncoder.encode_command("move", "kitchen"))
    print(DSLEncoder.encode_response_command("OK", "move", "kitchen"))
    print(DSLEncoder.encode_cloud())
    print(DSLEncoder.encode_error("Tool not found"))
    
    print("\nDecoding tests:")
    print(DSLDecoder.decode("S:Hello"))
    print(DSLDecoder.decode("T:math,2+2"))
    print(DSLDecoder.decode("C:move,kitchen"))
    print(DSLDecoder.decode("S:OK;C:move,kitchen"))
    print(DSLDecoder.decode("CL"))
    
    print("\nTool extraction:")
    print(DSLDecoder.extract_tool("T:math,2+2"))
    print(DSLDecoder.is_tool("T:math,2+2"))
    print(DSLDecoder.is_tool("S:Hello"))


from typing import Dict, List
import os
import json

def get_api_methods(oas_data: Dict) -> List[Dict]:
    """Get a list of API methods for genering utterances."""
    api_methods = []
    for path, methods in oas_data.get("paths", {}).items():
        for method_name, method_info in methods.items():
            api_method = {
                "api_name": oas_data.get("info", {}).get("title", ""),
                "api_description": oas_data.get("info", {}).get("description", ""),
                "api_method_name": method_name,
                "api_method_description": method_info.get("description", ""),
                "api_method_required_parameters": method_info.get("parameters", [])
            }
            api_methods.append(api_method)
    return api_methods

def get_tools(tool_path: str) -> List[Dict]:
    """Get a list of tools from the specified path."""
    tools = []
    
    for root, _, files in os.walk(tool_path):
        for file in files:
            if file.endswith(".json"):
                tool_path = os.path.join(root, file)
                with open(tool_path, "r") as f:
                    tool_data = json.load(f)
                    for tool in tool_data.get("api_list", []):
                        # name and description
                        name = tool.get("name", "")
                        description = tool.get("description", "")
                        
                        # parameters
                        properties = {}
                        required = []
                        for param in tool.get("parameters", []):
                            # getting info
                            param_name = param.get("name", "")
                            param_type = param.get("type", "").lower()
                            param_description = param.get("description", "")
                            param_required = param.get("required", False)
                            param_default = param.get("default", None)
                            param_values = param.get("values", None)

                            # required:
                            if param_required:
                                required.append(param_name)

                            # defining parameter dictionary
                            param_dict = {
                                "type": param_type,
                                "description": param_description}

                            if param_default is not None:
                                param_dict["default"] = param_default

                            if param_type == 'array' and param_values is not None:
                                param_dict["items"] = {"type": "string", "enum": param_values}
                            elif param_type != 'array' and param_values is not None: 
                                param_dict["enum"] = param_values
                                
                            # adding param_dict
                            properties[param_name] = param_dict

                        # adding new tool: 
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": description,
                                "parameters": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": required
                                }
                            }
                        })

    return tools
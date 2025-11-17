"""
API Documentation Standards for FinPilot VP-MAS

Provides standardized documentation generation and API specification
management for all agent endpoints.

Requirements: 9.4, 9.5
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .contracts import APIContracts


class DocumentationFormat(str, Enum):
    """Documentation output formats"""
    OPENAPI = "openapi"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class APIDocumentationStandards:
    """
    Standardized documentation generation for VP-MAS API endpoints.
    
    Provides consistent documentation across all agents with OpenAPI
    specification generation and interactive documentation.
    """
    
    @staticmethod
    def generate_openapi_spec() -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification for all agents"""
        
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "FinPilot VP-MAS API",
                "description": "Verifiable Planning Multi-Agent System API",
                "version": "1.0.0",
                "contact": {
                    "name": "FinPilot Development Team",
                    "email": "dev@finpilot.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "https://api.finpilot.com/v1",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.finpilot.com/v1", 
                    "description": "Staging server"
                },
                {
                    "url": "http://localhost:8000/v1",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            },
            "security": [
                {"bearerAuth": []},
                {"apiKey": []}
            ]
        }
        
        # Add paths from contracts
        contracts = APIContracts.get_all_contracts()
        
        for agent_type, agent_contracts in contracts.items():
            for endpoint_name, contract in agent_contracts.items():
                path = contract["path"]
                method = contract["method"].lower()
                
                if path not in spec["paths"]:
                    spec["paths"][path] = {}
                
                spec["paths"][path][method] = {
                    "summary": contract["description"],
                    "tags": [agent_type.replace("_", " ").title()],
                    "operationId": f"{agent_type}_{endpoint_name}",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": contract.get("request_schema", {})
                            }
                        }
                    } if method in ["post", "put", "patch"] else None,
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": contract.get("response_schema", {})
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/APIError"}
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/APIError"}
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/APIError"}
                                }
                            }
                        }
                    }
                }
        
        # Add common schemas
        spec["components"]["schemas"].update({
            "APIResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "object"},
                    "error": {"type": "string", "nullable": True},
                    "error_code": {"type": "string", "nullable": True},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string"},
                    "execution_time": {"type": "number"},
                    "api_version": {"type": "string"}
                },
                "required": ["success", "timestamp", "request_id", "execution_time", "api_version"]
            },
            "APIError": {
                "type": "object",
                "properties": {
                    "error_code": {"type": "string"},
                    "error_message": {"type": "string"},
                    "error_details": {"type": "object", "nullable": True},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string"},
                    "suggested_action": {"type": "string", "nullable": True}
                },
                "required": ["error_code", "error_message", "timestamp", "request_id"]
            }
        })
        
        return spec
    
    @staticmethod
    def generate_markdown_docs() -> str:
        """Generate Markdown documentation for all endpoints"""
        
        docs = """# FinPilot VP-MAS API Documentation

## Overview

The FinPilot Verifiable Planning Multi-Agent System (VP-MAS) provides a comprehensive API for financial planning, market analysis, and automated execution through a coordinated multi-agent architecture.

## Authentication

All API endpoints require authentication using either:
- Bearer token (JWT): `Authorization: Bearer <token>`
- API Key: `X-API-Key: <api_key>`

## Base URLs

- Production: `https://api.finpilot.com/v1`
- Staging: `https://staging-api.finpilot.com/v1`
- Development: `http://localhost:8000/v1`

## Response Format

All API responses follow a standardized format:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "error_code": null,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345",
  "execution_time": 0.25,
  "api_version": "v1"
}
```

## Error Handling

Error responses include detailed information:

```json
{
  "success": false,
  "error": "Request format is invalid",
  "error_code": "INVALID_REQUEST",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345",
  "suggested_action": "Check request schema"
}
```

## Agents and Endpoints

"""
        
        contracts = APIContracts.get_all_contracts()
        
        for agent_type, agent_contracts in contracts.items():
            agent_title = agent_type.replace("_", " ").title()
            docs += f"\n### {agent_title} Agent\n\n"
            
            for endpoint_name, contract in agent_contracts.items():
                docs += f"#### {contract['method']} {contract['path']}\n\n"
                docs += f"{contract['description']}\n\n"
                
                if "request_schema" in contract:
                    docs += "**Request Schema:**\n```json\n"
                    docs += str(contract["request_schema"]) + "\n```\n\n"
                
                if "response_schema" in contract:
                    docs += "**Response Schema:**\n```json\n"
                    docs += str(contract["response_schema"]) + "\n```\n\n"
                
                if "error_codes" in contract:
                    docs += "**Error Codes:**\n"
                    for code, description in contract["error_codes"].items():
                        docs += f"- `{code}`: {description}\n"
                    docs += "\n"
        
        return docs
    
    @staticmethod
    def generate_postman_collection() -> Dict[str, Any]:
        """Generate Postman collection for API testing"""
        
        collection = {
            "info": {
                "name": "FinPilot VP-MAS API",
                "description": "Comprehensive API collection for FinPilot multi-agent system",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{auth_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "http://localhost:8000/v1",
                    "type": "string"
                },
                {
                    "key": "auth_token",
                    "value": "",
                    "type": "string"
                }
            ],
            "item": []
        }
        
        contracts = APIContracts.get_all_contracts()
        
        for agent_type, agent_contracts in contracts.items():
            agent_folder = {
                "name": agent_type.replace("_", " ").title(),
                "item": []
            }
            
            for endpoint_name, contract in agent_contracts.items():
                request_item = {
                    "name": contract["description"],
                    "request": {
                        "method": contract["method"],
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "url": {
                            "raw": "{{base_url}}" + contract["path"],
                            "host": ["{{base_url}}"],
                            "path": contract["path"].split("/")[1:]
                        }
                    }
                }
                
                if contract["method"] in ["POST", "PUT", "PATCH"]:
                    request_item["request"]["body"] = {
                        "mode": "raw",
                        "raw": "{\n  // Add request body based on schema\n}",
                        "options": {
                            "raw": {
                                "language": "json"
                            }
                        }
                    }
                
                agent_folder["item"].append(request_item)
            
            collection["item"].append(agent_folder)
        
        return collection


# Documentation generation utilities
def export_documentation(format_type: DocumentationFormat, output_path: str = None) -> str:
    """Export API documentation in specified format"""
    
    if format_type == DocumentationFormat.OPENAPI:
        content = APIDocumentationStandards.generate_openapi_spec()
        import json
        return json.dumps(content, indent=2)
    
    elif format_type == DocumentationFormat.MARKDOWN:
        return APIDocumentationStandards.generate_markdown_docs()
    
    elif format_type == DocumentationFormat.JSON:
        content = APIDocumentationStandards.generate_postman_collection()
        import json
        return json.dumps(content, indent=2)
    
    else:
        raise ValueError(f"Unsupported documentation format: {format_type}")


# CLI utility for documentation generation
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate FinPilot API documentation")
    parser.add_argument("--format", choices=["openapi", "markdown", "json"], 
                       default="markdown", help="Documentation format")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        content = export_documentation(DocumentationFormat(args.format), args.output)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(content)
            print(f"Documentation exported to {args.output}")
        else:
            print(content)
    
    except Exception as e:
        print(f"Error generating documentation: {e}", file=sys.stderr)
        sys.exit(1)
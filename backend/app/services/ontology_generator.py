"""
Ontology Generation Service
Analyzes document text and generates entity and relationship type definitions
suitable for social media opinion simulation
"""

import json
import logging
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# Ontology generation system prompt
ONTOLOGY_SYSTEM_PROMPT = """You are an expert in knowledge graph ontology design. Your task is to analyze document content and simulation requirements, then design entity types and relationship types suitable for **social media opinion simulation systems**.

**IMPORTANT: You must output valid JSON format only. Do not output any other content.**

## Core Task Background

We are building a **social media opinion simulation system**. In this system:
- Each entity is an "account" or "actor" that can voice opinions, interact, and spread information on social media
- Entities influence each other through comments, reposts, and responses
- We need to simulate how different stakeholders react and spread information during opinion events

Therefore, **entities must be real-world actors who can voice opinions on social media**:

**Valid entity types include**:
- Specific individuals (public figures, involved parties, opinion leaders, experts, ordinary people)
- Companies and enterprises (including official accounts)
- Organizations (universities, associations, NGOs, unions, etc.)
- Government departments and regulatory agencies
- Media institutions (newspapers, TV stations, self-media, websites)
- Social media platforms themselves
- Group representatives (alumni associations, fan groups, advocacy groups, etc.)

**Invalid entity types include**:
- Abstract concepts (e.g., "public opinion", "sentiment", "trends")
- Topics/subjects (e.g., "academic integrity", "education reform")
- Viewpoints/attitudes (e.g., "supporters", "opponents")

## Output Format

Output JSON with the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Brief description (English, max 100 characters)",
            "attributes": [
                {
                    "name": "Attribute name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationship type name (English, UPPER_SNAKE_CASE)",
            "description": "Brief description (English, max 100 characters)",
            "source_targets": [
                {"source": "Source entity type", "target": "Target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of document content and key stakeholders"
}
```

## Design Guidelines (CRITICAL!)

### 1. Entity Type Design - MUST FOLLOW STRICTLY

**Quantity requirement: EXACTLY 10 entity types**

**Hierarchy requirement (must include both specific and fallback types)**:

Your 10 entity types must include:

A. **Fallback types (MUST include, place as last 2)**:
   - `Person`: Fallback type for any individual person not fitting other specific person types
   - `Organization`: Fallback type for any organization not fitting other specific organization types

B. **Specific types (8 types, designed based on document content)**:
   - Design specific types based on main actors mentioned in documents
   - Example: For academic events: `Student`, `Professor`, `University`
   - Example: For business events: `Company`, `CEO`, `Employee`

**Why fallback types are needed**:
- Documents mention people like "teachers", "bystanders", "anonymous users"
- Without specific type matches, they should be categorized as `Person`
- Similarly, small organizations and temporary groups should be `Organization`

**Specific type design principles**:
- Identify high-frequency or key actor types from the document
- Each specific type should have clear boundaries, no overlap
- Description must clarify distinction between the specific type and fallback type

### 2. Relationship Type Design

- Quantity: 6-10 types
- Relationships should reflect real interactions on social media
- Ensure source_targets cover defined entity types

### 3. Attribute Design

- 1-3 key attributes per entity type
- **IMPORTANT**: Do not use reserved names: `name`, `uuid`, `group_id`, `created_at`, `summary`
- Recommended: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Reference Entity Types

**Specific Person Types**:
- Student, Professor, Journalist, Celebrity, Executive, Official, Lawyer, Doctor

**Specific Organization Types**:
- University, Company, GovernmentAgency, MediaOutlet, Hospital, School, NGO

**Fallback Types**:
- Person, Organization

## Reference Relationship Types

- WORKS_FOR, STUDIES_AT, AFFILIATED_WITH, REPRESENTS, REGULATES
- REPORTS_ON, COMMENTS_ON, RESPONDS_TO, SUPPORTS, OPPOSES
- COLLABORATES_WITH, COMPETES_WITH
"""


class OntologyGenerator:
    """
    Ontology Generator
    Analyzes document content and generates entity and relationship type definitions
    """
    
    # Maximum text length to send to LLM (50,000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition
        
        Args:
            document_texts: List of document text strings
            simulation_requirement: Description of simulation requirements
            additional_context: Optional additional context
            
        Returns:
            Dictionary containing entity_types, edge_types, and analysis_summary
            
        Raises:
            ValueError: If LLM returns invalid JSON or API fails
        """
        try:
            # Build user message
            user_message = self._build_user_message(
                document_texts, 
                simulation_requirement,
                additional_context
            )
            
            messages = [
                {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            # Call LLM and parse response
            self.logger.info("Calling LLM to generate ontology...")
            result = self.llm_client.chat_json(
                messages=messages,
                temperature=0.3,
                max_tokens=4096
            )
            
            # Validate and post-process result
            result = self._validate_and_process(result)
            
            self.logger.info(
                f"Ontology generation complete: {len(result.get('entity_types', []))} "
                f"entity types, {len(result.get('edge_types', []))} relationship types"
            )
            
            return result
            
        except ValueError as e:
            self.logger.error(f"JSON parsing error from LLM: {str(e)}")
            raise ValueError(f"Failed to parse ontology from LLM response: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating ontology: {str(e)}")
            raise
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message for LLM"""
        
        # Combine all document texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Truncate if exceeds max length (only for LLM, doesn't affect graph building)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += (
                f"\n\n...(Original document: {original_length} characters, "
                f"truncated to first {self.MAX_TEXT_LENGTH_FOR_LLM} for analysis)..."
            )
        
        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Additional Context

{additional_context}
"""
        
        message += """
Based on the above content, design entity types and relationship types suitable for social media opinion simulation.

**MANDATORY RULES**:
1. Output EXACTLY 10 entity types
2. The last 2 must be fallback types: Person and Organization
3. First 8 are specific types designed based on document content
4. All entity types must be real-world actors capable of voicing opinions on social media
5. Do not use reserved attribute names (name, uuid, group_id, created_at, summary)
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process LLM result"""
        
        # Ensure required fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Validate entity types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description doesn't exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Validate relationship types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API limitations: max 10 entity types, max 10 relationship types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous user"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Check if fallback types already exist
        entity_names = {e.get("name", "") for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Collect fallback types to add
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Remove existing types if adding fallbacks would exceed limit
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from end (keep more important specific types at beginning)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)
        
        # Final defensive check to ensure we don't exceed limits
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)


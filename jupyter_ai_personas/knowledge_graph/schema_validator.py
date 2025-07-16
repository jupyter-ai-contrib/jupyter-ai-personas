from neo4j import GraphDatabase


class SchemaValidator:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def get_actual_schema(self):
        """Get the actual schema from Neo4j database"""
        with self.driver.session() as session:
            # node labels and their properties
            node_result = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect(propertyName) as properties
            """)

            # relationship types
            rel_result = session.run("""
                CALL db.schema.relTypeProperties()
                YIELD relType
                RETURN DISTINCT relType
            """)

            # relationship patterns
            pattern_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] as from_label, type(r) as rel_type, labels(b)[0] as to_label
                LIMIT 20
            """)

            nodes = {record["nodeType"]: record["properties"] for record in node_result}
            relationships = [record["relType"] for record in rel_result]
            patterns = [
                (record["from_label"], record["rel_type"], record["to_label"])
                for record in pattern_result
            ]

            return {
                "nodes": nodes,
                "relationships": relationships,
                "patterns": patterns,
            }

    def generate_schema_info(self):
        """Generate schema information string for agents"""
        schema = self.get_actual_schema()

        info = "ACTUAL DATABASE SCHEMA:\n\n"

        # Node types and properties
        info += "NODES:\n"
        for node_type, properties in schema["nodes"].items():
            info += f"- {node_type}: properties {{{', '.join(properties)}}}\n"

        # Relationships
        info += f"\nRELATIONSHIPS:\n"
        for rel in schema["relationships"]:
            info += f"- {rel}\n"

        # Relationship patterns
        info += f"\nVALID PATTERNS:\n"
        for from_label, rel_type, to_label in schema["patterns"]:
            info += f"- ({from_label})-[:{rel_type}]->({to_label})\n"

        info += self._get_sample_files()

        # examples
        info += f"\nEXAMPLE QUERIES:\n"
        if schema["patterns"]:
            pattern = schema["patterns"][0]
            info += f"- MATCH ({pattern[0].lower()}:{pattern[0]})-[:{pattern[1]}]->({pattern[2].lower()}:{pattern[2]}) RETURN {pattern[0].lower()}.name\n"

        return info

    def _get_sample_files(self):
        """Get sample files in the database"""
        with self.driver.session() as session:
            # Python files (Class/Function nodes with 'file' property)
            py_result = session.run("""
                MATCH (n)
                WHERE n.file IS NOT NULL
                RETURN DISTINCT n.file as file, labels(n) as labels
                LIMIT 5
            """)

            # Check for other files (File nodes with 'path' property)
            file_result = session.run("""
                MATCH (f:File)
                RETURN DISTINCT f.path as file, f.type as type
                LIMIT 5
            """)

            info = "\nFILES IN DATABASE:\n"

            py_files = list(py_result)
            other_files = list(file_result)

            if py_files:
                info += "Python files (Class/Function nodes):\n"
                for record in py_files:
                    info += f"- {record['file']} ({', '.join(record['labels'])})\n"

            if other_files:
                info += "Other files (File nodes):\n"
                for record in other_files:
                    info += f"- {record['file']} ({record['type']})\n"

            if not py_files and not other_files:
                info += "- No files found in database\n"

            info += "\nQUERY PATTERNS:\n"
            info += "- For Python: MATCH (n:Class) WHERE n.file CONTAINS 'filename' RETURN n.name\n"
            info += "- For Other files: MATCH (f:File) WHERE f.path CONTAINS 'filename' RETURN f.path\n"

            return info

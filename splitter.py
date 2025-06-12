import logging
from elasticsearch import Elasticsearch
from typing import Dict


class Splitter:
    """Text splitter based on regex patterns similar to jina_text_segmenter"""
    
    def __init__(self, splitter_id="jina_text_splitter", chunk_size: int = 512, chunk_overlap: int = 16):
        self.splitter_id = splitter_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def get_script_source(self) -> str:
        # Define variables for magic numbers
        MAX_HEADING_LENGTH = 7
        MAX_HEADING_CONTENT_LENGTH = 200
        MAX_HEADING_UNDERLINE_LENGTH = 200
        MAX_HTML_HEADING_ATTRIBUTES_LENGTH = 100
        MAX_LIST_ITEM_LENGTH = 200
        MAX_NESTED_LIST_ITEMS = 6
        MAX_LIST_INDENT_SPACES = 7
        MAX_BLOCKQUOTE_LINE_LENGTH = 200
        MAX_BLOCKQUOTE_LINES = 15
        MAX_CODE_BLOCK_LENGTH = 1500
        MAX_CODE_LANGUAGE_LENGTH = 20
        MAX_INDENTED_CODE_LINES = 20
        MAX_TABLE_CELL_LENGTH = 200
        MAX_TABLE_ROWS = 20
        MAX_HTML_TABLE_LENGTH = 2000
        MIN_HORIZONTAL_RULE_LENGTH = 3
        MAX_SENTENCE_LENGTH = 400
        MAX_QUOTED_TEXT_LENGTH = 300
        MAX_PARENTHETICAL_CONTENT_LENGTH = 200
        MAX_NESTED_PARENTHESES = 5
        MAX_MATH_INLINE_LENGTH = 100
        MAX_MATH_BLOCK_LENGTH = 500
        MAX_PARAGRAPH_LENGTH = 1000
        MAX_STANDALONE_LINE_LENGTH = 800
        MAX_HTML_TAG_ATTRIBUTES_LENGTH = 100
        MAX_HTML_TAG_CONTENT_LENGTH = 1000
        LOOKAHEAD_RANGE = 100

        AVOID_AT_START = r'[\s\]})>,\']'
        # 注意：移除了不支持的Unicode属性，改为基本字符
        PUNCTUATION = r'[.!?…]|\.{3}|[\u2026\u2047-\u2049]'
        QUOTE_END = r"(?:'(?=`)|''(?=``))"
        SENTENCE_END = f"(?:{PUNCTUATION}(?<!{AVOID_AT_START}(?={PUNCTUATION}))|{QUOTE_END})(?=\\S|$)"
        SENTENCE_BOUNDARY = f"(?:{SENTENCE_END}|(?=[\\r\\n]|$))"
        LOOKAHEAD_PATTERN = f"(?:(?!{SENTENCE_END}).){{{1},{LOOKAHEAD_RANGE}}}{SENTENCE_END}"
        NOT_PUNCTUATION_SPACE = f"(?!{PUNCTUATION}\\s)"

        def _get_sentence_pattern(max_length):
            """Generate sentence pattern with specific max length"""
            return f"{NOT_PUNCTUATION_SPACE}(?:[^\\r\\n]{{1,{max_length}}}{SENTENCE_BOUNDARY}|[^\\r\\n]{{1,{max_length}}}(?={PUNCTUATION}|{QUOTE_END})(?:{LOOKAHEAD_PATTERN})?){AVOID_AT_START}*"
        
        pattern = "|".join([
            # 1. Headings (Setext-style, Markdown, and HTML-style, with length constraints)
            f"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}|\\w[^\\r\\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\\r?\\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>)[^\\r\\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}(?:<\\/h[1-6]>)?(?:\\r?\\n|$))",
            
            # New pattern for citations
            f"(?:\\[[0-9]+\\][^\\r\\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})",
            
            # 2. List items (bulleted, numbered, lettered, or task lists, including nested, up to three levels, with length constraints)
            f"(?:(?:^|\\r?\\n)[ \\t]{{0,3}}(?:[-*+•]|\\d{{1,3}}\\.\\w\\.|\\[[ xX]\\])[ \\t]+{_get_sentence_pattern(MAX_LIST_ITEM_LENGTH)}" +
            f"(?:(?:\\r?\\n[ \\t]{{2,5}}(?:[-*+•]|\\d{{1,3}}\\.\\w\\.|\\[[ xX]\\])[ \\t]+{_get_sentence_pattern(MAX_LIST_ITEM_LENGTH)}){{0,{MAX_NESTED_LIST_ITEMS}}}" +
            f"(?:\\r?\\n[ \\t]{{4,{MAX_LIST_INDENT_SPACES}}}(?:[-*+•]|\\d{{1,3}}\\.\\w\\.|\\[[ xX]\\])[ \\t]+{_get_sentence_pattern(MAX_LIST_ITEM_LENGTH)}){{0,{MAX_NESTED_LIST_ITEMS}}})?)",
            
            # 3. Block quotes (including nested quotes and citations, up to three levels, with length constraints)
            f"(?:(?:^>(?:>|\\s{{2,}}){{0,2}}{_get_sentence_pattern(MAX_BLOCKQUOTE_LINE_LENGTH)}\\r?\\n?){{1,{MAX_BLOCKQUOTE_LINES}}})",
            
            # 4. Code blocks (fenced, indented, or HTML pre/code tags, with length constraints)
            f"(?:(?:^|\\r?\\n)(?:```|~~~)(?:\\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\\r?\\n[\\s\\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:```|~~~)\\r?\\n?" +
            f"|(?:(?:^|\\r?\\n)(?: {{4}}|\\t)[^\\r\\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\\r?\\n(?: {{4}}|\\t)[^\\r\\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\\r?\\n?)" +
            f"|(?:<pre>(?:<code>)?[\\s\\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:<\\/code>)?<\\/pre>))",
            
            # 5. Tables (Markdown, grid tables, and HTML tables, with length constraints)
            f"(?:(?:^|\\r?\\n)(?:\\|[^\\r\\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\\|(?:\\r?\\n\\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\\|){{0,1}}(?:\\r?\\n\\|[^\\r\\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\\|){{0,{MAX_TABLE_ROWS}}}" +
            f"|<table>[\\s\\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?<\\/table>))",
            
            # 6. Horizontal rules (Markdown and HTML hr tag)
            f"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\\s*$|<hr\\s*\\/?>)",
            
            # 10. Standalone lines or phrases (including single-line blocks and HTML elements, with length constraints)
            f"(?!{AVOID_AT_START})(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>)?{_get_sentence_pattern(MAX_STANDALONE_LINE_LENGTH)}(?:<\\/[a-zA-Z]+>)?(?:\\r?\\n|$))",
            
            # 7. Sentences or phrases ending with punctuation (including ellipsis and Unicode punctuation)
            f"(?!{AVOID_AT_START}){_get_sentence_pattern(MAX_SENTENCE_LENGTH)}",
            
            # 8. Quoted text, parenthetical phrases, or bracketed content (with length constraints)
            "(?:" +
            f"(?<!\\w)\"\"\"[^\"]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\"\"\"(?!\\w)" +
            f"|(?<!\\w)(?:['\"`'\"])[^\\r\\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\\1(?!\\w)" +
            f"|(?<!\\w)`[^\\r\\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}'(?!\\w)" +
            f"|(?<!\\w)``[^\\r\\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}''(?!\\w)" +
            f"|\\([^\\r\\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\\([^\\r\\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\\)[^\\r\\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\\)" +
            f"|\\[[^\\r\\n\\[\\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\\[[^\\r\\n\\[\\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\\][^\\r\\n\\[\\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\\]" +
            f"|\\$[^\\r\\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\\$" +
            f"|`[^`\\r\\n]{{0,{MAX_MATH_INLINE_LENGTH}}}`" +
            ")",
            
            # 9. Paragraphs (with length constraints)
            f"(?!{AVOID_AT_START})(?:(?:^|\\r?\\n\\r?\\n)(?:<p>)?{_get_sentence_pattern(MAX_PARAGRAPH_LENGTH)}(?:<\\/p>)?(?=\\r?\\n\\r?\\n|$))",            
            
            # 11. HTML-like tags and their content (including self-closing tags and attributes, with length constraints)
            f"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:[\\s\\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}?<\\/[a-zA-Z]+>|\\s*\\/>))",
            
            # 12. LaTeX-style math expressions (inline and block, with length constraints)
            f"(?:(?:\\$\\$[\\s\\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\\$\\$)|(?:\\$[^\\$\\r\\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\\$))",
            
            # 14. Fallback for any remaining content (with length constraints)
            f"(?!{AVOID_AT_START}){_get_sentence_pattern(MAX_STANDALONE_LINE_LENGTH)}"
        ])

        # print('pattern', pattern)

        """Return Painless script implementing text segmentation"""
        return """
        // Ensure ctx, attachment, and content are valid
        if (ctx == null || ctx.attachment == null || ctx.attachment.content == null) {
            if (ctx != null) { ctx.chunks = new ArrayList(); } // Ensure chunks is always initialized
            return; 
        }
        String content = ctx.attachment.content;

        // Ensure params and config are valid
        if (params == null || params.splitter_config == null) {
            if (ctx != null) { ctx.chunks = new ArrayList(); }
            // It's good practice to set an error field or log if possible,
            // but for now, just return with empty chunks.
            return;
        }
        Map config = params.splitter_config;

        // Ensure chunk_size is a number and positive
        if (!(config.get('chunk_size') instanceof Number)) {
             if (ctx != null) { ctx.chunks = new ArrayList(); }
            return;
        }
        int chunkSize = ((Number)config.chunk_size).intValue();
        if (chunkSize <= 0) {
            if (ctx != null) { ctx.chunks = new ArrayList(); }
            return;
        }

        List chunks = new ArrayList();
        String text = content.trim();

        if (text.length() == 0) {
            ctx.chunks = chunks;
            return;
        }

        Matcher matcher = /(""" + pattern + """)/m.matcher(text);
        for (int i = 0; matcher.find(); i++) {
            String part = matcher.group();
            Map chunkData = new HashMap();
            chunkData.put("content", part);
            chunkData.put("index", i);
            chunkData.put("length", part.length());
            chunkData.put("start", matcher.start());
            chunkData.put("end", matcher.end());
            chunks.add(chunkData);
        }
        
        ctx.chunks = chunks;
        """

    def debug_script(self, es_client: Elasticsearch, test_content: str = None):
        """Debug the Painless script using ES debug API"""
        if test_content is None:
            test_content = "这是测试文本。包含多个句子！"
        
        # PARAGRAPH_PATTERN = f"(?:(?:^|\\r?\\n\\r?\\n)(?![-*+>\\s]|\\d+\\.)[^\\r\\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?:\\r?\\n(?!\\r?\\n|[-*+>\\s]|\\d+\\.)[^\\r\\n]{{1,{MAX_PARAGRAPH_LENGTH}}})*)(?:\\r?\\n\\r?\\n|$)"

        # 10. Standalone lines or phrases (including single-line blocks and HTML elements, with length constraints)
        # (?![\\s\\]})>,'])(?:^(?:<[a-zA-Z][^>]{0,100}>)?(?![.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}]\\s)(?:[^\\r\\n]{1,800}(?:(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}](?<![\\s\\]})>,'](?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}]))|(?:'(?=`)|''(?=``)))(?=\\S|$)|(?=[\\r\\n]|$))|[^\\r\\n]{1,800}(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}]|(?:'(?=`)|''(?=``)))(?:(?:(?!(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}](?<![\\s\\]})>,'](?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}]))|(?:'(?=`)|''(?=``)))(?=\\S|$)).){1,100}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}](?<![\\s\\]})>,'](?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}]))|(?:'(?=`)|''(?=``)))(?=\\S|$))?)[\\s\\]})>,']*(?:</[a-zA-Z]+>)?(?:\\r?\\n|$))

        try:
            # 1. 首先测试基本脚本语法
            print("=== Testing basic script compilation ===")
            basic_test = es_client.scripts_painless_execute(
                body={
                    "script": {
                        "source": """
                        String content = params.test_content;
                        Map result = new HashMap();
                        result.put("text", content);
                        result.put("status", "ok");
                        result.put("length", content.length());
                        return result;
                        """,
                        "params": {
                            "test_content": test_content
                        }
                    }
                },
                error_trace=True,
            )
            print("Basic script test passed:", basic_test)
            
        except Exception as e:
            logging.error(e)
            print(f"Basic script test failed: {e}")
            return False
        
        try:
            # 2. 测试最简单的 ingest 脚本
            print("\n=== Testing simplest ingest script ===")
            simplest_script = """
            if (ctx.attachment != null && ctx.attachment.content != null) {
                ctx.debug_info = "Content found";
                ctx.content_length = ctx.attachment.content.length();
            }
            """
            
            simple_test = es_client.ingest.simulate(
                body={
                    "pipeline": {
                        "processors": [{
                            "script": {
                                "source": simplest_script
                            }
                        }]
                    },
                    "docs": [{
                        "_source": {
                            "attachment": {
                                "content": test_content
                            }
                        }
                    }]
                }
            )
            print("Simplest ingest script test passed")
            
        except Exception as e:
            print(f"Simplest ingest script test failed: {e}")
            return False
        
        try:
            # 3. 测试基本的分割逻辑
            print("\n=== Testing basic splitting logic ===")
            basic_split_script = """
            if (ctx.attachment != null && ctx.attachment.content != null) {
                String content = ctx.attachment.content;
                List chunks = new ArrayList();
                
                String text = content.trim();
                int chunkSize = 50;
                
                for (int i = 0; i < text.length(); i += chunkSize) {
                    int end = (int)Math.min(i + chunkSize, text.length());
                    String chunk = text.substring(i, end);
                    chunks.add(chunk);
                }
                
                ctx.chunks = chunks;
            }
            """
            
            basic_test = es_client.ingest.simulate(
                body={
                    "pipeline": {
                        "processors": [{
                            "script": {
                                "source": basic_split_script
                            }
                        }]
                    },
                    "docs": [{
                        "_source": {
                            "attachment": {
                                "content": test_content
                            }
                        }
                    }]
                }
            )
            
            doc = basic_test['docs'][0]['doc']
            if 'chunks' in doc:
                print(f"Basic splitting test passed! Created {len(doc['chunks'])} chunks")
            else:
                print("Basic splitting test failed - no chunks created")
                print("Document keys:", list(doc.keys()))
            
        except Exception as e:
            print(f"Basic splitting test failed: {e}")
            return False
        
        try:
            # 4. 测试完整的分割脚本
            print("\n=== Testing full splitting script ===")
            full_test = es_client.ingest.simulate(
                body={
                    "pipeline": {
                        "processors": [{
                            "script": {
                                "source": self.get_script_source(),
                                "params": {
                                    "splitter_config": {
                                        "chunk_size": self.chunk_size,
                                        "chunk_overlap": self.chunk_overlap
                                    }
                                }
                            }
                        }]
                    },
                    "docs": [{
                        "_source": {
                            "attachment": {
                                "content": test_content
                            }
                        }
                    }]
                }
            )
            
            # 检查结果
            print(full_test)
            doc = full_test['docs'][0]['doc']
            if 'chunks' in doc:
                print(f"Full script test passed! Created {len(doc['chunks'])} chunks")
                for i, chunk in enumerate(doc['chunks'][:2]):  # 只显示前两个
                    print(f"  Chunk {i+1}: {str(chunk)[:50]}...")
            else:
                print("Full script test passed but no chunks created")
                print("Document keys:", list(doc.keys()))
            
            return True
            
        except Exception as e:
            print(f"Full script test failed: {e}")
            return False

    def get_processor(self) -> Dict:
        """Return the processor configuration for the pipeline"""
        return {
            "script": {
                "id": self.splitter_id,
                "params": {
                    "splitter_config": {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    }
                }
            }
        }

    def init_script(self, es_client: Elasticsearch, force_recreate: bool = False):
        """Initialize the script in Elasticsearch"""
        if not force_recreate:
            try:
                existing = es_client.get_script(id=self.splitter_id)
                if existing:
                    logging.debug(f"Splitter script already exists: {self.splitter_id}")
                    return
            except:
                pass
        
        try:
            es_client.put_script(
                id=self.splitter_id,
                body={
                    "script": {
                        "lang": "painless",
                        "source": self.get_script_source(),
                    }
                }
            )
            logging.debug(f"Splitter script initialized successfully: {self.splitter_id}")
        except Exception as e:
            logging.warning(f"Splitter script initialization failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage with improved testing
    es = Elasticsearch("http://0.0.0.0:9200").options(ignore_status=[400])
    splitter = Splitter("jina_text_splitter", chunk_size=200, chunk_overlap=20)
    
    # Test with simpler text first
    with open("examples/zh.txt", "r", encoding="utf-8") as f:
        test_text = f.read()
    
    # Debug the script first
    print("=== Debugging Painless Script ===")
    if not splitter.debug_script(es, test_text):
        print("Script debugging failed, exiting...")
        exit(1)
    
    # Test pipeline simulation
    try:
        print("\n=== Testing Pipeline Simulation ===")
        response = es.ingest.simulate(
            body={
                "pipeline": {
                    "description": "Text splitting pipeline",
                    "processors": [{
                        "script": {
                            "source": splitter.get_script_source(),
                            "lang": "painless",
                            "params": {
                                "splitter_config": {
                                    "chunk_size": splitter.chunk_size,
                                    "chunk_overlap": splitter.chunk_overlap
                                }
                            }
                        }
                    }]
                },
                "docs": [
                    {
                        "_source": {
                            "attachment": {
                                "content": test_text
                            }
                        }
                    }
                ]
            },
        )
        
        # Display results
        doc = response['docs'][0]['doc']['_source']
        if 'chunks' in doc:
            print(f"Successfully split into {len(doc['chunks'])} chunks:")
            for i, chunk in enumerate(doc['chunks']):
                print(f"\nChunk {i+1}:")
                if isinstance(chunk, dict):
                    print(f"Content: {chunk.get('content', chunk)}")
                    if 'metadata' in chunk:
                        print(f"Metadata: {chunk['metadata']}")
                else:
                    print(f"Content: {chunk}")
        else:
            print("No chunks created")
            print("Document content:", doc)
            
    except Exception as e:
        print(f"Pipeline simulation failed: {e}")

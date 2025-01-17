from book import ContentType

class Model:
    def make_text_prompt(self, text: str, source_language, target_language: str) -> str:
        return f"从{source_language}翻译为{target_language}：{text}"

    def make_table_prompt(self, table: str, source_language, target_language: str) -> str:
        return f"从{source_language}翻译为{target_language}，保持间距（空格，分隔符），以表格形式返回：\n{table}"

    def translate_prompt(self, content, source_language:str, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, source_language, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), source_language, target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")

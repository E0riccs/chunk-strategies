from src.utils.logger import setup_logger

class ChunkLoader:
    """
        支持从外部导入已分割好的数据
    """
    def __init__(self, config):
        """
            Args:
                config (dict): The configuration for the special chunk strategy.
        """
        self.logger = setup_logger(__name__)
        self.config = config
    
    def _libsql_load(self):
        """
            从libsql数据库中加载已分割好的数据
        """
        pass
    
    def load(self):
        """
            从文件中加载已分割好的数据
        """
        if self.config['source_type'] == 'libsql':
            return self._libsql_load()
        else:
            self.logger.error(f"Chunking strategy '{self.config['name']}' is not supported yet, you need define it firstly.")
            raise ValueError("Error: Chunking strategy '{self.config['name']}' is not supported yet, you need define it firstly.")
        

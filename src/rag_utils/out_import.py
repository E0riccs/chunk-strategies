from src.utils.logger import setup_logger
import sqlite3

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
        读取已经切割好的 chunks，依赖于 Cherry Studio 中 的TS libsql 具体实现。
        
        Returns:
            list: 包含所有pageContent的列表
        
        Raises:
            sqlite3.Error: 当数据库操作失败时抛出
            KeyError: 当配置中缺少source_path时抛出
        """
        try:
            db_file_path = self.config['source_path']
            vectors_dict = {}
            
            with sqlite3.connect(db_file_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON") # 外键
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM vectors")
                # 获取列名
                column_names = [description[0] for description in cursor.description]
                # 将所有数据加载到字典中
                for row in cursor.fetchall():
                    row_dict = {column_names[i]: row[i] for i in range(len(column_names))}
                    vectors_dict[row[0]] = row_dict  # 使用id作为键
                
                self.logger.info(f"Successfully loaded {len(vectors_dict)} vectors from database")
                
                # 从字典中提取pageContent列表
                page_contents = []
                for vector_id, vector_data in vectors_dict.items():
                    if vector_data.get('pageContent'):
                        page_contents.append(vector_data['pageContent'])
                
                self.logger.info(f"Extracted {len(page_contents)} page contents from vectors")
                return page_contents
                
        except KeyError as e:
            self.logger.error(f"Missing required configuration key: {e}")
            raise
        except sqlite3.Error as e:
            self.logger.error(f"Database error occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during database loading: {e}")
            raise
        
    def load(self):
        """
            从文件中加载已分割好的数据
        """
        if self.config['source_type'] == 'libsql':
            return self._libsql_load()
        else:
            self.logger.error(f"Chunking strategy '{self.config['name']}' is not supported yet, you need define it firstly.")
            raise ValueError("Error: Chunking strategy '{self.config['name']}' is not supported yet, you need define it firstly.")
        

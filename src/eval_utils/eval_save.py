"""
    对一批次的实验结果进行最终整合、统计、储存。
"""
import pandas as pd
import datetime
import os

from src.utils.logger import setup_logger

from src.eval_utils.eval_model import EvalResponseModel


class EvalSaver:
    def __init__(self, results_dir: str, experiments_config: dict):
        self.logger = setup_logger(__name__)
        self.results_dir = results_dir
        self.experiments_config = experiments_config

        # 批次开始时的时间戳
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 需要记录的数据名称
        self.configured_cols_order = []
        category_cols = self.experiments_config.get('result_columns_order', {})
        for col in category_cols.values():
            if isinstance(col, list):
                for col_detail in col:
                    if isinstance(col_detail, dict):
                        col_name = next(iter(col_detail))
                        self.configured_cols_order.append(col_name)
            else: # dict
                col_name = next(iter(col))
                self.configured_cols_order.append(col_name) 
        
        # 储存文件
        self.summary_dir = os.path.join(self.results_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True) # 确保目录存在
        self.summary_filepath = os.path.join(self.summary_dir, f"experiment_results_{self.timestamp}.csv")
        if os.path.exists(self.summary_filepath):
            try:
                self.df = pd.read_csv(self.summary_filepath)
            except pd.errors.EmptyDataError: # 空文件
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()
        
    def _format(self, entry: dict):
        """
            转化格式为 文本、数字 等 可以加入 df 的格式
        """
        pass
        

    def _statistics(self, entry: dict):
        """
            根据 EvalDetailModel 统计计算 得到result
        """
        # 1. llm_score_ratio
        eval_response = entry.pop('eval_metrics', None)
        if eval_response:
            # cosine_similaritys, llm_scores
            cosine_similaritys = []
            llm_scores = []
            for metrics in eval_response:
                cosine_similaritys.append(metrics.cosine_similarity)
                llm_scores.append(metrics.llm_score)

            entry['cosine_similaritys'] = cosine_similaritys
            entry['llm_evaluation_scores'] = llm_scores

            # llm_score_ratio
            max_sum_score = len(eval_response) * 2 # define in Prompt
            sum_score = 0
            for metrics in eval_response:
                sum_score += metrics.llm_score
            entry['llm_score_ratio'] = sum_score / max_sum_score
        
        
        return entry


    def add_one_result(self, **kwargs):
        """
            构造一个字典，代表一次实验的完整结果，计算相关指标后调用保存方法。
            使用 kwargs 接收所有实验结果字段。
        """
        result_entry = {}
        result_entry.update(kwargs)
        
        result_entry = self._statistics(result_entry)
        self.append_and_save_entry(result_entry)

        file_type_name = kwargs.get('file_type_name', 'Unknown File Type')
        chunking_strategy_name = kwargs.get('chunking_strategy_name', 'Unknown Strategy')
        self.logger.info(f"Saved result for {file_type_name} with {chunking_strategy_name} to {self.summary_filepath}")


        

    def append_and_save_entry(self, entry: dict):
        """
            追加单个实验结果到DataFrame，并保存整个DataFrame到CSV。
            如果 entry 缺少实验预定义的列，则抛出 ValueError。
            如果 entry 包含更多的列，则这些额外列也会被保存。
        """
        if not entry:
            self.logger.info("No experiment result to save.")
            return

        # 检查是否所有 configured_cols 在 entry 中都存在
        if self.configured_cols_order:
            missing_cols = [col for col in self.configured_cols_order if col not in entry.keys()]
            if missing_cols:
                raise ValueError(f"Missing required columns in experiment result: {', '.join(missing_cols)}. Required by experiment setting.")
        
        new_row_df = pd.DataFrame([entry])

        # 将新行追加到现有的 self.df
        if self.df.empty:
            self.df = new_row_df
        else:
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # 代码中可能加入设置中不存在的项目，对整个 self.df 进行列排序，确保兼容性
        df_to_save = self.df
        current_df_columns = list(self.df.columns)

        # 存在于 DataFrame 中的按其顺序排列
        ordered_present_cols = [col for col in self.configured_cols_order if col in current_df_columns]
        
        # 不存在的合并在后
        additional_cols = [col for col in current_df_columns if col not in self.configured_cols_order]
        final_columns_for_saving = ordered_present_cols + additional_cols
        df_to_save = self.df[final_columns_for_saving]
        
        try:
            df_to_save.to_csv(self.summary_filepath, index=False, encoding='utf-8-sig')
        except IOError as e:
            self.logger.error(f"Error saving summary CSV file {self.summary_filepath}: {e}")
 
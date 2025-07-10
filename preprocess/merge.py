import os
import glob
import pandas as pd

def merge_csv_files(input_folder, output_file):
    """
    合并多个CSV文件，只保留一个标题行
    """
    # 获取文件夹中所有的CSV文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    # 检查是否找到CSV文件
    if not csv_files:
        print(f"在文件夹 {input_folder} 中没有找到CSV文件")
        return
    
    # 读取并合并CSV文件
    dfs = []
    for i, csv_file in enumerate(csv_files):
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 如果是第一个文件，保留标题行；否则跳过标题行
        if i == 0:
            dfs.append(df)
        else:
            dfs.append(df.iloc[1:])
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的CSV文件
    merged_df.to_csv(output_file, index=False)
    print(f"成功合并 {len(csv_files)} 个CSV文件到 {output_file}")

if __name__ == "__main__":
    # 设置输入文件夹和输出文件路径
    input_folder = "your_chat_history_path"
    output_file = "output_path"
    
    # 调用合并函数
    merge_csv_files(input_folder, output_file)

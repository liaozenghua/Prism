import json
import pandas as pd
import openpyxl


def replace_examples_in_excel(json_file_path, excel_file_path, output_file_path=None):
    """
    用JSON文件中的内容替换Excel文件中的example列

    参数:
    json_file_path: JSON文件路径
    excel_file_path: Excel文件路径
    output_file_path: 输出文件路径，如果为None则覆盖原文件
    """

    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        print(f"从JSON文件读取到 {len(json_data)} 条数据")

        # 读取Excel文件
        df = pd.read_excel(excel_file_path)

        print(f"Excel文件有 {len(df)} 行数据")
        print(f"Excel文件列名: {list(df.columns)}")

        # 查找example列（忽略大小写）
        example_col = None
        for col in df.columns:
            if col.lower() == 'example':
                example_col = col
                break

        if example_col is None:
            print("错误：在Excel文件中未找到'example'列")
            return False

        # 确定要替换的行数
        rows_to_replace = min(len(df), len(json_data))

        # 替换example列的内容
        for i in range(rows_to_replace):
            df.loc[i, example_col] = json_data[i]

        print(f"已替换 {rows_to_replace} 行的example列内容")

        # 如果JSON数据比Excel行数多，提醒用户
        if len(json_data) > len(df):
            print(f"注意：JSON文件有 {len(json_data)} 条数据，但Excel只有 {len(df)} 行，只替换了前 {len(df)} 行")

        # 如果Excel行数比JSON数据多，提醒用户
        if len(df) > len(json_data):
            print(f"注意：Excel有 {len(df)} 行，但JSON只有 {len(json_data)} 条数据，剩余行的example列保持不变")

        # 保存文件
        output_path = output_file_path if output_file_path else excel_file_path
        df.to_excel(output_path, index=False)

        print(f"文件已保存到: {output_path}")
        return True

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON文件格式错误: {e}")
        return False
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 文件路径
    json_file = "/home/data/liaozenghua/ReST-SDLG/scenario/example_428.json"
    excel_file = "/home/data/liaozenghua/ReST-SDLG/scenario/labelled.xlsx"
    output_file = "/home/data/liaozenghua/ReST-SDLG/scenario/labelled-1.xlsx"  # 可选：指定输出文件名，如果不指定则覆盖原文件

    # 执行替换操作
    success = replace_examples_in_excel(json_file, excel_file, output_file)

    if success:
        print("操作完成！")
    else:
        print("操作失败，请检查错误信息。")
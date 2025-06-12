import json
import os

import numpy as np
from openai import OpenAI
import time
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("DMXAPI_API_KEY"))

class DeepResearchReportEvaluator:
    """面向DeepResearch场景的长报告评估系统"""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://www.dmxapi.com/v1"):
        """
        初始化评估器

        Args:
            api_key: API密钥
            model: 评估使用的模型
            base_url: API基础URL
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.dimensions = self._get_evaluation_dimensions()

    def _get_evaluation_dimensions(self) -> Dict[str, Dict]:
        """
        获取评估维度及其权重

        Returns:
            Dict: 评估维度及其权重和描述
        """
        return {
            "factual_accuracy": {
                "weight": 0.20,
                "description": "内容的事实准确性，包括数据、引用、理论和概念的正确性",
                "prompt_template": """
                作为一位专业的内容评估专家，请仔细评估以下内容的事实准确性。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 数据和统计的准确性
                2. 引用和参考的正确性
                3. 理论和概念的准确表述
                4. 事实陈述的可验证性
                5. 避免错误信息或误导性内容

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告中存在的具体事实错误或准确之处。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["具体的准确或错误点1", "具体的准确或错误点2", ...]
                }}
                """
            },
            "depth_of_analysis": {
                "weight": 0.15,
                "description": "分析深度，包括对主题的深入探讨、多角度思考和批判性思维",
                "prompt_template": """
                作为一位深度研究分析专家，请评估以下报告内容的分析深度。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 研究深度与广度
                2. 多角度思考与批判性分析
                3. 洞察力与创新性
                4. 问题复杂性的把握
                5. 证据链与论证强度

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告分析深度的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["分析深度的优点或缺点1", "分析深度的优点或缺点2", ...]
                }}
                """
            },
            "logical_coherence": {
                "weight": 0.15,
                "description": "逻辑连贯性，包括论证的逻辑性、因果关系的合理性和论述的一致性",
                "prompt_template": """
                作为一位逻辑分析专家，请评估以下报告内容的逻辑连贯性。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 论证的逻辑结构
                2. 因果关系的合理性
                3. 论述的一致性与连贯性
                4. 推理过程的有效性
                5. 避免逻辑谬误

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告中的逻辑优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["逻辑连贯性的优点或缺点1", "逻辑连贯性的优点或缺点2", ...]
                }}
                """
            },
            "structural_organization": {
                "weight": 0.10,
                "description": "结构组织，评估报告的层次结构、段落安排和信息流转的合理性",
                "prompt_template": """
                作为一位文档结构专家，请评估以下报告内容的结构组织。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 整体结构的合理性
                2. 章节与段落的组织
                3. 信息流转的顺畅度
                4. 层次结构的清晰度
                5. 标题与内容的一致性

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告结构的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["结构组织的优点或缺点1", "结构组织的优点或缺点2", ...]
                }}
                """
            },
            "comprehensiveness": {
                "weight": 0.10,
                "description": "内容全面性，评估报告对问题各方面的覆盖程度和完整性",
                "prompt_template": """
                作为一位研究全面性评估专家，请评估以下报告内容的全面性。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 问题各方面的覆盖程度 
                2. 关键要素的完整性
                3. 相关背景信息的提供
                4. 不同观点的呈现
                5. 潜在影响和未来发展的讨论

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告全面性的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["全面性的优点或缺点1", "全面性的优点或缺点2", ...]
                }}
                """
            },
            "language_quality": {
                "weight": 0.10,
                "description": "语言质量，包括表达的清晰性、专业性和流畅性",
                "prompt_template": """
                作为一位语言表达专家，请评估以下报告内容的语言质量。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 表达的清晰度与简洁性
                2. 专业术语的准确使用
                3. 语法与拼写的正确性
                4. 句式多样性与流畅度
                5. 语言风格的一致性与适当性

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告语言质量的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["语言质量的优点或缺点1", "语言质量的优点或缺点2", ...]
                }}
                """
            },
            "relevance": {
                "weight": 0.10,
                "description": "相关性，评估报告内容与问题的相关度和针对性",
                "prompt_template": """
                作为一位内容相关性专家，请评估以下报告内容与问题的相关性。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 内容与问题的直接相关性
                2. 问题核心要素的回应程度
                3. 避免离题和不相关内容
                4. 针对性解决方案或见解的提供
                5. 对问题的实质性贡献

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告相关性的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["相关性的优点或缺点1", "相关性的优点或缺点2", ...]
                }}
                """
            },
            "originality": {
                "weight": 0.10,
                "description": "原创性和创新性，评估报告提供的新颖见解、创新思路和原创贡献",
                "prompt_template": """
                作为一位创新思维评估专家，请评估以下报告内容的原创性和创新性。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 新颖见解与独特观点
                2. 创新思路与方法
                3. 超越常规思维的程度
                4. 独特价值的创造
                5. 区别于现有研究/观点的程度

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告原创性的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["原创性的优点或缺点1", "原创性的优点或缺点2", ...]
                }}
                """
            },
            "practical_value": {
                "weight": 0.10,
                "description": "实用价值，评估报告内容的应用价值、可操作性和实际意义",
                "prompt_template": """
                作为一位实用价值评估专家，请评估以下报告内容的实用价值。

                评估问题: {query}

                报告内容:
                {content}

                请从以下几个方面进行评估:
                1. 应用价值与实际意义
                2. 解决方案的可行性与可操作性
                3. 对实际问题的指导作用
                4. 成本效益与实施难度
                5. 长期价值与可持续性

                请给出1-10的评分（10分为最高，1分为最低），并详细说明评分理由，指出报告实用价值的优缺点。

                输出格式:
                {{
                    "score": 你的评分（1-10的整数）,
                    "reasoning": "你的评分理由",
                    "highlights": ["实用价值的优点或缺点1", "实用价值的优点或缺点2", ...]
                }}
                """
            }
        }

    def evaluate_dimension(self, query: str, content: str, dimension: str) -> Dict:
        """
        评估特定维度

        Args:
            query: 评估问题
            content: 报告内容
            dimension: 评估维度

        Returns:
            Dict: 评估结果
        """
        if dimension not in self.dimensions:
            raise ValueError(f"未知的评估维度: {dimension}")

        dimension_info = self.dimensions[dimension]
        prompt = dimension_info["prompt_template"].format(
            query=query,
            content=content
        )

        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位专业的内容评估专家，需要对研究报告进行评估。请严格按照要求的格式输出JSON结果。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model,
                    response_format={"type": "json_object"}
                )

                # 提取JSON响应
                result_text = response.choices[0].message.content
                result = json.loads(result_text)

                # 添加维度信息
                result["dimension"] = dimension
                result["dimension_description"] = dimension_info["description"]
                result["weight"] = dimension_info["weight"]

                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"评估维度 {dimension} 时出错: {e}，正在重试...")
                    time.sleep(2)  # 稍等片刻再重试
                else:
                    print(f"评估维度 {dimension} 失败: {e}")
                    # 返回默认评分
                    return {
                        "dimension": dimension,
                        "dimension_description": dimension_info["description"],
                        "weight": dimension_info["weight"],
                        "score": 5,  # 默认中等分数
                        "reasoning": f"评估过程中出现错误: {str(e)}",
                        "highlights": ["评估失败"]
                    }

    def evaluate_report(self, query: str, content: str) -> Dict:
        """
        全面评估报告

        Args:
            query: 评估问题
            content: 报告内容

        Returns:
            Dict: 评估结果
        """
        # 分段处理超长内容
        max_tokens = 15000  # 根据模型限制调整
        content_chunks = self._split_content(content, max_tokens)

        results = {}
        overall_score = 0.0
        total_weight = 0.0

        # 对每个维度进行评估
        for dimension, info in self.dimensions.items():
            # 如果内容过长，需要分段评估并汇总
            if len(content_chunks) > 1:
                dimension_score = 0
                dimension_reasoning = []
                dimension_highlights = []

                for i, chunk in enumerate(content_chunks):
                    print(f"评估维度 {dimension} 的内容块 {i + 1}/{len(content_chunks)}...")
                    chunk_result = self.evaluate_dimension(query, chunk, dimension)
                    dimension_score += chunk_result["score"]
                    dimension_reasoning.append(f"块{i + 1}: {chunk_result['reasoning']}")
                    dimension_highlights.extend(chunk_result["highlights"])

                # 计算平均分
                dimension_score /= len(content_chunks)
                dimension_result = {
                    "dimension": dimension,
                    "dimension_description": info["description"],
                    "weight": info["weight"],
                    "score": round(dimension_score, 1),
                    "reasoning": " ".join(dimension_reasoning),
                    "highlights": dimension_highlights[:5]  # 限制亮点数量
                }
            else:
                print(f"评估维度: {dimension}...")
                dimension_result = self.evaluate_dimension(query, content, dimension)

            results[dimension] = dimension_result
            weighted_score = dimension_result["score"] * info["weight"]
            overall_score += weighted_score
            total_weight += info["weight"]

        # 计算加权平均分
        if total_weight > 0:
            overall_score = overall_score / total_weight

        # 生成总体评估
        meta_evaluation = self._generate_meta_evaluation(query, content, results)

        return {
            "query": query,
            "dimensions": results,
            "overall_score": round(overall_score, 2),
            "meta_evaluation": meta_evaluation
        }

    def _split_content(self, content: str, max_tokens: int) -> List[str]:
        """
        将内容分割成多个块

        Args:
            content: 完整内容
            max_tokens: 每块最大token数量

        Returns:
            List[str]: 内容块列表
        """
        # 简单估算token数量：平均每个单词1.3个token
        if len(content) < max_tokens * 4:  # 粗略估计: 1个token ~= 4个字符
            return [content]

        # 按段落分割
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < max_tokens * 4:
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _generate_meta_evaluation(self, query: str, content: str, dimension_results: Dict) -> Dict:
        """
        生成总体评估

        Args:
            query: 评估问题
            content: 报告内容
            dimension_results: 各维度评估结果

        Returns:
            Dict: 总体评估结果
        """
        # 准备维度结果摘要
        dimensions_summary = ""
        for dim, result in dimension_results.items():
            dimensions_summary += f"{dim} ({result['score']}分): {result['reasoning'][:200]}...\n\n"

        prompt = f"""
        作为一位资深研究评估专家，请基于以下各维度的评估结果，对整篇研究报告进行总体评价。

        评估问题: {query}

        各维度评估结果摘要:
        {dimensions_summary}

        请提供:
        1. 报告的主要优势（3-5点）
        2. 报告的主要不足（3-5点）
        3. 改进建议（3-5点）
        4. 总体评价（200字以内）

        输出格式:
        {{
            "strengths": ["优势1", "优势2", ...],
            "weaknesses": ["不足1", "不足2", ...],
            "improvement_suggestions": ["建议1", "建议2", ...],
            "overall_comment": "总体评价"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的研究报告评估专家，需要基于各维度评估结果对报告进行总体评价。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except Exception as e:
            print(f"生成总体评估时出错: {e}")
            return {
                "strengths": ["无法生成优势评估"],
                "weaknesses": ["无法生成不足评估"],
                "improvement_suggestions": ["无法生成改进建议"],
                "overall_comment": f"评估过程中出现错误: {str(e)}"
            }

    def generate_visual_report(self, evaluation_result: Dict) -> str:
        """
        生成可视化评估报告（Markdown格式）

        Args:
            evaluation_result: 评估结果

        Returns:
            str: Markdown格式的评估报告
        """
        query = evaluation_result["query"]
        overall_score = evaluation_result["overall_score"]
        dimensions = evaluation_result["dimensions"]
        meta = evaluation_result["meta_evaluation"]

        # 构建Markdown报告
        report = [
            f"# DeepResearch长报告评估结果\n",
            f"## 评估问题\n\n{query}\n",
            f"## 总体评分: {overall_score}/10\n",
            f"## 总体评价\n\n{meta['overall_comment']}\n",
            f"## 各维度评分\n"
        ]

        # 各维度评分表格
        report.append("| 评估维度 | 权重 | 得分 |\n|---------|------|------|\n")
        for dim_name, dim_result in dimensions.items():
            report.append(f"| {dim_name} | {dim_result['weight']} | {dim_result['score']} |\n")

        report.append("\n## 详细评估\n")

        # 各维度详细评估
        for dim_name, dim_result in dimensions.items():
            report.append(f"### {dim_name} ({dim_result['score']}/10)\n\n")
            report.append(f"**描述**: {dim_result['dimension_description']}\n\n")
            report.append(f"**评价**: {dim_result['reasoning']}\n\n")

            report.append("**亮点**:\n")
            for highlight in dim_result['highlights']:
                report.append(f"- {highlight}\n")
            report.append("\n")

        # 优势与不足
        report.append("## 主要优势\n\n")
        for strength in meta["strengths"]:
            report.append(f"- {strength}\n")

        report.append("\n## 主要不足\n\n")
        for weakness in meta["weaknesses"]:
            report.append(f"- {weakness}\n")

        report.append("\n## 改进建议\n\n")
        for suggestion in meta["improvement_suggestions"]:
            report.append(f"- {suggestion}\n")

        return "".join(report)

    def batch_evaluate(self, query_report_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        批量评估多个报告

        Args:
            query_report_pairs: 问题-报告对列表

        Returns:
            List[Dict]: 评估结果列表
        """
        results = []
        for i, (query, report) in enumerate(query_report_pairs):
            print(f"评估报告 {i + 1}/{len(query_report_pairs)}...")
            result = self.evaluate_report(query, report)
            results.append(result)

            # 防止API限流
            if i < len(query_report_pairs) - 1:
                time.sleep(1)

        return results

    def compare_reports(self, query: str, reports: Dict[str, str]) -> Dict:
        """
        比较多个模型的报告

        Args:
            query: 评估问题
            reports: 模型名称-报告内容字典

        Returns:
            Dict: 比较结果
        """
        # 首先评估每个报告
        results = {}
        for model_name, report in reports.items():
            print(f"评估模型 {model_name} 的报告...")
            results[model_name] = self.evaluate_report(query, report)

        # 准备比较提示
        models_summary = ""
        for model_name, result in results.items():
            models_summary += f"{model_name} (总分: {result['overall_score']})\n"
            # 添加每个维度的得分
            for dim, dim_result in result['dimensions'].items():
                models_summary += f"- {dim}: {dim_result['score']}\n"
            models_summary += "\n"

        comparison_prompt = f"""
        作为一位模型评估专家，请对以下多个模型生成的研究报告进行比较分析。

        研究问题: {query}

        各模型评估总结:
        {models_summary}

        请提供:
        1. 各模型的相对优势与劣势比较
        2. 不同模型在各维度上的表现对比
        3. 最佳模型推荐及理由
        4. 模型能力差异的关键洞察

        输出格式:
        {{
            "model_comparison": "详细的模型比较分析",
            "dimension_comparison": {{
                "维度1": "各模型在此维度的比较",
                "维度2": "各模型在此维度的比较",
                ...
            }},
            "best_model": "最佳模型名称",
            "best_model_reason": "最佳模型推荐理由",
            "key_insights": ["洞察1", "洞察2", ...]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的研究报告评估专家，需要对多个模型生成的报告进行比较分析。"
                    },
                    {
                        "role": "user",
                        "content": comparison_prompt
                    }
                ],
                model=self.model,
                response_format={"type": "json_object"}
            )

            comparison_result = json.loads(response.choices[0].message.content)

            # 构建完整的比较结果
            return {
                "query": query,
                "individual_results": results,
                "comparison": comparison_result
            }
        except Exception as e:
            print(f"生成比较分析时出错: {e}")
            return {
                "query": query,
                "individual_results": results,
                "comparison": {
                    "model_comparison": f"比较分析生成失败: {str(e)}",
                    "dimension_comparison": {},
                    "best_model": "未能确定",
                    "best_model_reason": "比较分析生成失败",
                    "key_insights": ["分析失败"]
                }
            }


# 示例使用
def evaluate_single_report():
    """评估单个报告示例"""
    evaluator = DeepResearchReportEvaluator(
        api_key=os.getenv("DMXAPI_API_KEY"),
        base_url=os.getenv("DMXAPI_BASE_URL"),
        model=os.getenv("DMXAPI_MODEL"),

        # api_key=os.getenv("VOLCENGINE_API_KEY"),
        # base_url=os.getenv("VOLCENGINE_BASE_URL"),
        # model=os.getenv("VOLCENGINE_MODEL"),
    )

    # query = "请分析人工智能在医疗领域的应用现状与未来发展趋势"
    query = "大型语言模型(LLM)最新研究进展综述：预训练、微调、应用与评估"
    # report_content = """
    # # 人工智能在医疗领域的应用现状与未来发展趋势
    #
    # ## 引言
    #
    # 人工智能(AI)技术在医疗领域的应用正经历前所未有的发展。随着深度学习、自然语言处理和计算机视觉等技术的进步，AI已经从概念验证阶段迈向临床应用。本报告将全面分析AI在医疗领域的应用现状，探讨面临的挑战，并预测未来发展趋势。
    #
    # ## 当前应用现状
    #
    # ### 1. 医学影像分析
    #
    # 医学影像分析是AI应用最成熟的领域之一。深度学习算法在放射学、病理学和皮肤科等多个科室显示出与专家水平相当甚至超越的性能。例如:
    #
    # - **放射学**: FDA已批准多款AI辅助诊断系统用于X光、CT和MRI图像分析，如用于乳腺癌筛查的算法准确率达到95%以上
    # - **病理学**: 数字病理切片分析算法能够自动识别肿瘤细胞，准确率超过90%
    # - **皮肤科**: AI已能够区分良性和恶性皮肤病变，在黑色素瘤检测上表现尤为突出
    #
    # ### 2. 临床决策支持系统
    #
    # 基于AI的临床决策支持系统(CDSS)正逐渐进入医疗实践:
    #
    # - **诊断辅助**: 整合电子健康记录(EHR)数据，提供疾病诊断建议和风险评估
    # - **治疗方案推荐**: 基于最新研究和患者数据，推荐个性化治疗方案
    # - **药物相互作用警告**: 自动检测潜在的药物相互作用风险
    #
    # ### 3. 健康管理与监测
    #
    # AI技术在健康管理领域应用广泛:
    #
    # - **远程监护**: 利用可穿戴设备数据实时监测患者生命体征
    # - **慢性病管理**: 通过智能算法预测糖尿病、心脏病等慢性病风险和病情变化
    # - **行为健康**: AI聊天机器人提供心理健康初步评估和支持
    #
    # ### 4. 药物研发
    #
    # AI技术显著加速了药物研发过程:
    #
    # - **靶点发现**: 通过分析大规模生物数据识别潜在治疗靶点
    # - **分子设计**: 生成式AI设计潜在活性分子结构
    # - **临床试验优化**: 预测药物反应，优化患者选择和试验设计
    #
    # ## 面临的挑战
    #
    # 尽管AI在医疗领域取得了显著进展，但仍面临多重挑战:
    #
    # ### 1. 数据挑战
    #
    # - **数据质量与标准化**: 医疗数据往往不完整、有噪声且缺乏标准化
    # - **隐私与安全**: 在利用数据训练模型的同时确保患者隐私保护
    # - **数据偏见**: 训练数据中的人口学偏差可能导致AI系统表现不公平
    #
    # ### 2. 技术挑战
    #
    # - **模型可解释性**: 医疗决策要求高度透明，但深度学习模型通常是"黑盒"
    # - **泛化能力**: 在新数据和不同环境中保持稳定性能
    # - **实时性能**: 某些临床场景要求极高的实时处理能力
    #
    # ### 3. 临床整合挑战
    #
    # - **工作流程整合**: 将AI工具无缝整合到现有医疗工作流程
    # - **医生接受度**: 临床医生对AI系统的信任和采用程度
    # - **监管审批**: 满足严格的医疗器械监管要求
    #
    # ## 未来发展趋势
    #
    # ### 1. 多模态AI系统
    #
    # 未来的医疗AI将整合多种数据源:
    #
    # - 结合影像、基因组、电子健康记录和可穿戴设备数据
    # - 全面分析患者状况，提供更精准的诊断和治疗建议
    #
    # ### 2. 联邦学习与差分隐私
    #
    # 解决数据隐私问题的技术进步:
    #
    # - 联邦学习允许在保护数据隐私的同时进行分布式模型训练
    # - 差分隐私技术确保个体数据不被逆向推导
    #
    # ### 3. 可解释AI与知识图谱
    #
    # 提高AI系统透明度:
    #
    # - 可解释AI技术将揭示决策过程背后的逻辑
    # - 医学知识图谱将提供推理依据和证据支持
    #
    # ### 4. AI辅助精准医疗
    #
    # 个性化医疗的革命:
    #
    # - 基于个体基因组数据的疾病风险预测
    # - 个性化治疗方案推荐与药物剂量优化
    # - 患者特异性预后预测
    #
    # ### 5. 人机协作新模式
    #
    # 重新定义医患关系:
    #
    # - AI承担重复性任务，医生专注于复杂决策和患者关怀
    # - 增强医生与患者之间的沟通和理解
    #
    # ## 结论
    #
    # 人工智能在医疗领域已经从概念验证阶段迈向实际应用，正在改变医疗服务的提供方式。尽管面临数据、技术和监管等挑战，但AI技术的持续发展将推动医疗服务向更精准、个性化、高效和普惠的方向发展。未来，AI不仅是辅助工具，还将成为医疗系统的核心组成部分，医疗从业者需要积极适应这一变革，共同塑造AI驱动的医疗未来。
    # """
    with open("samples4/report.md","r",encoding="utf-8") as f:
        report_content=f.read()
    # 执行评估
    evaluation_result = evaluator.evaluate_report(query, report_content)

    # 生成视觉化报告
    visual_report = evaluator.generate_visual_report(evaluation_result)

    # 打印评估结果
    print("评估完成！总体评分:", evaluation_result["overall_score"])
    print("\n视觉化报告预览:")
    print(visual_report[:500] + "...")
    print(visual_report)
    return evaluation_result, visual_report


def compare_multiple_reports():
    """比较多个模型报告示例"""
    evaluator = DeepResearchReportEvaluator(
        api_key=os.getenv("DMXAPI_API_KEY"),
        model=os.getenv("DMXAPI_MODEL","gpt-4o"),
        base_url=os.getenv("DMXAPI_BASE_URL"),

        # api_key=os.getenv("VOLCENGINE_API_KEY"),
        # base_url=os.getenv("VOLCENGINE_BASE_URL"),
        # model=os.getenv("VOLCENGINE_MODEL"),
    )

    query = "分析区块链技术在供应链管理中的应用前景"

    # 不同模型生成的报告
    reports = {
        "Model_A": "# 区块链技术在供应链管理中的应用前景分析\n\n## 1. 引言\n\n区块链作为一种分布式账本技术...",
        "Model_B": "# 区块链与供应链管理的融合\n\n近年来，区块链技术因其去中心化、不可篡改、透明可追溯等特性...",
        "Model_C": "# 供应链区块链应用前景报告\n\n## 摘要\n\n本报告探讨了区块链技术在现代供应链管理中的应用潜力..."
    }

    # 执行比较分析
    comparison_result = evaluator.compare_reports(query, reports)

    # 打印比较结果
    print("比较分析完成！")
    print("最佳模型:", comparison_result["comparison"]["best_model"])
    print("最佳模型推荐理由:", comparison_result["comparison"]["best_model_reason"])

    return comparison_result


def main():
    """主函数"""
    print("DeepResearch长报告评估系统演示")
    print("=" * 50)

    # 单个报告评估
    print("\n1. 单个报告评估示例")
    result, report = evaluate_single_report()
    #
    # 多个报告比较
    print("\n2. 多个报告比较示例")
    comparison = compare_multiple_reports()

    print("\n评估系统演示完成")


if __name__ == "__main__":
    main()
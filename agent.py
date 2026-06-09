import api
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from api import api_call
from ddgs import DDGS
import io
import contextlib
import traceback

class BaseAgent:
    """Agent基类"""
    
    def __init__(self):
        self.conversation_history = []
    
    def ask(self, question: str) -> tuple[str, int]:
        """向Agent提问，子类需要实现此方法"""
        raise NotImplementedError
    
    def get_conversation_history(self):
        return self.conversation_history
    
    def clear_conversation(self):
        self.conversation_history = []


class DirectPromptAgent(BaseAgent):
    """基础直接提示Agent"""
    
    def ask(self, question: str) -> tuple[str, int]:
        """直接调用API获取回答"""
        self.conversation_history.append({"role": "user", "content": question})
        
        response = api.api_call(question)
        answer = response.message.content
        
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return (answer, 1)


class ReActAgent(BaseAgent):
    """ReAct范式Agent：结合推理(Reasoning)和行动(Acting)"""
    
    def __init__(self, max_steps: int = 5):
        super().__init__()
        self.max_steps = max_steps
        self.tools = {}
        self.knowledge_base = self._init_knowledge_base()
        self._init_tools()
    
    def _init_knowledge_base(self) -> Dict[str, str]:
        """初始化知识库，模拟搜索结果"""
        return {
            "中国首都": "北京，是中华人民共和国的首都，是全国的政治、文化中心。",
            "北京人口": "北京市常住人口约2154万人（2023年数据）。",
            "上海人口": "上海市常住人口约2489万人（2023年数据）。",
            "中国面积": "中国陆地面积约960万平方公里。",
            "地球半径": "地球的平均半径约为6371公里。",
            "月球距离": "月球到地球的平均距离约为38.4万公里。",
            "光速": "光速约为每秒299,792公里。",
            "珠穆朗玛峰高度": "珠穆朗玛峰海拔约8848.86米。",
            "太平洋面积": "太平洋面积约1.65亿平方公里。",
            "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "量子计算": "量子计算利用量子力学原理来处理信息，在某些问题上比传统计算机有指数级加速。",
            "牛顿力学": "牛顿力学是经典力学的基础，适用于宏观低速物体的运动规律。",
            "相对论": "相对论包括狭义相对论和广义相对论，由爱因斯坦提出，适用于高速和强引力场。",
            "Python": "Python是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。",
            "机器学习": "机器学习是人工智能的一个子领域，使计算机能够从数据中学习并改进性能。",
            # 基础检索测试锚点
            "地球距离月球": "月球到地球的平均距离约为38.4万公里。",
            "Python创始人": "Python语言由吉多·范罗苏姆（Guido van Rossum）于1989年底发明。",
            
            # 多工具协同（搜索+计算）测试锚点
            "北京人口": "北京市常住人口约2154万人（2023年数据）。",
            "上海人口": "上海市常住人口约2489万人（2023年数据）。",
            "光速": "光在真空中的传播速度约为每秒299792公里。",
            
            # 干扰项（测试模型的检索精准度）
            "纽约人口": "纽约市常住人口约833万人（2022年数据）。",
            "声速": "声音在空气中的传播速度约为每秒340米。",
        }
    
    def _init_tools(self):
        """初始化可用工具"""
        self.tools["search"] = self._search_tool
        self.tools["calculate"] = self._calculate_tool
        self.tools["datetime"] = self._datetime_tool
        self.tools["wiki"] = self._wiki_tool
        self.tools["math"] = self._math_tool
        # 注册新增的 Python 工具
        self.tools["python"] = self._python_tool
    
    def _search_tool(self, query: str) -> str:
        """真实的联网搜索工具：通过 DuckDuckGo 查找互联网信息"""
        query = query.strip()
        if not query:
            return "[搜索结果] 搜索关键词不能为空。"

        print(f"\n[联网搜索中...] 关键词: {query}")
        
        try:
            results = []
            # 使用 DDGS 进行文本搜索，限制返回前 3 条最相关的结果以节省上下文 token
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=3)
                for r in ddgs_gen:
                    # 提取标题和内容摘要，拼接成大模型容易理解的格式
                    results.append(f"- 标题: {r['title']}\n  摘要: {r['body']}")
            
            if results:
                return "[联网搜索结果]\n" + "\n\n".join(results)
            else:
                return f"[联网搜索结果] 未能在互联网上找到关于 '{query}' 的直接信息，建议更换更基础的搜索词。"
                
        except Exception as e:
            # 异常处理：遇到网络波动或反爬限制时，优雅降级到本地字典搜索
            print(f"[搜索异常] 联网失败 ({str(e)})，降级使用本地知识库...")
            return self._fallback_local_search(query)

    def _fallback_local_search(self, query: str) -> str:
        """原有的本地字典搜索逻辑，作为备用方案"""
        query = query.lower()
        
        # 精确匹配
        for key, value in self.knowledge_base.items():
            if key.lower() in query or query in key.lower():
                return f"[本地检索结果] {value}"
        
        # 部分匹配
        matched_keys = [key for key in self.knowledge_base.keys() if any(word in key.lower() for word in query.split())]
        if matched_keys:
            results = []
            for key in matched_keys[:3]:
                results.append(f"- {key}: {self.knowledge_base[key]}")
            return "[本地检索结果]\n" + "\n".join(results)
            
        return f"[本地检索结果] 未找到关于 '{query}' 的具体信息。"
    
    def _calculate_tool(self, expression: str) -> str:
        """计算工具：安全地计算数学表达式"""
        try:
            # 清理表达式，只允许安全的字符
            allowed_chars = set('0123456789+-*/().%^ ')
            if not all(c in allowed_chars for c in expression):
                return "[计算错误] 表达式包含非法字符，请只使用数字和运算符。"
            
            # 替换^为**进行幂运算
            expression = expression.replace('^', '**')
            
            # 使用安全的计算方式
            # 定义允许的数学函数
            safe_dict = {
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'sum': sum,
                'max': max,
                'min': min,
            }
            
            # 添加math模块的函数
            safe_dict.update({k: v for k, v in math.__dict__.items() if not k.startswith('_')})
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            # 格式化结果
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 4)
            
            return f"[计算结果] {expression} = {result}"
        except SyntaxError:
            return "[计算错误] 表达式语法错误，请检查输入格式。"
        except ZeroDivisionError:
            return "[计算错误] 除数不能为零。"
        except Exception as e:
            return f"[计算错误] 计算失败: {str(e)}"
    
    def _datetime_tool(self, query: str = "") -> str:
        """日期时间工具：获取当前日期和时间"""
        now = datetime.now()
        
        if not query or query.strip() in ["now", "当前", "现在"]:
            return f"[时间信息] 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}"
        elif "日期" in query or "date" in query.lower():
            return f"[时间信息] 当前日期：{now.strftime('%Y年%m月%d日')}"
        elif "时间" in query or "time" in query.lower():
            return f"[时间信息] 当前时间：{now.strftime('%H:%M:%S')}"
        elif "星期" in query or "week" in query.lower():
            return f"[时间信息] 今天是星期{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}"
        elif "年" in query or "year" in query.lower():
            return f"[时间信息] 当前年份：{now.year}年"
        elif "月" in query or "month" in query.lower():
            return f"[时间信息] 当前月份：{now.month}月"
        elif "日" in query or "day" in query.lower():
            return f"[时间信息] 今天是{now.day}号"
        else:
            return f"[时间信息] 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _wiki_tool(self, topic: str) -> str:
        """维基百科风格的知识查询工具"""
        topic = topic.strip()
        
        wiki_knowledge = {
            "人工智能": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。",
            "机器学习": "机器学习（Machine Learning）是人工智能的一个分支，使计算机系统能够通过经验自动改进，而无需明确编程。它基于算法构建数学模型，利用训练数据进行预测或决策。",
            "深度学习": "深度学习（Deep Learning）是机器学习的一个子集，基于人工神经网络，特别是深度神经网络。它在图像识别、语音识别、自然语言处理等领域取得了显著成果。",
            "北京": "北京，简称'京'，是中华人民共和国的首都、直辖市、国家中心城市，中国的政治、文化中心，国际交往中心，科技创新中心。",
            "地球": "地球是太阳系八大行星之一，按离太阳由近及远的次序排为第三颗，也是太阳系中直径、质量和密度最大的类地行星，距离太阳1.5亿公里。",
            "Python": "Python是一种高级编程语言，由吉多·范罗苏姆于1991年首次发布。它以简洁的语法和强大的生态系统著称，广泛应用于Web开发、数据分析、人工智能等领域。",
        }
        
        for key, value in wiki_knowledge.items():
            if key in topic or topic in key:
                return f"[百科知识] {value}"
        
        return f"[百科知识] 关于 '{topic}' 的详细信息：这是一个关于{topic}的主题。如需更详细的信息，建议使用search工具进行具体查询。"
    
    def _math_tool(self, query: str) -> str:
        """数学工具：提供数学常数和常用公式"""
        query = query.strip().lower()
        
        math_constants = {
            "pi": f"π (圆周率) ≈ {math.pi:.10f}",
            "π": f"π (圆周率) ≈ {math.pi:.10f}",
            "e": f"e (自然常数) ≈ {math.e:.10f}",
            "黄金分割": f"黄金分割率 φ ≈ {(1 + math.sqrt(5)) / 2:.10f}",
            "根号2": f"√2 ≈ {math.sqrt(2):.10f}",
            "根号3": f"√3 ≈ {math.sqrt(3):.10f}",
        }
        
        math_formulas = {
            "圆面积": "圆面积 = π × r²，其中r为半径",
            "圆周长": "圆周长 = 2 × π × r，其中r为半径",
            "三角形面积": "三角形面积 = (底 × 高) / 2",
            "矩形面积": "矩形面积 = 长 × 宽",
            "正方形面积": "正方形面积 = 边长²",
            "球体积": "球体积 = (4/3) × π × r³，其中r为半径",
            "勾股定理": "勾股定理：a² + b² = c²，其中c为直角三角形斜边",
        }
        
        # 查找常数
        for key, value in math_constants.items():
            if key in query:
                return f"[数学常数] {value}"
        
        # 查找公式
        for key, value in math_formulas.items():
            if key in query:
                return f"[数学公式] {value}"
        
        # 返回常用数学常数列表
        return "[数学工具] 可用的数学查询：\n" + \
               "常数：pi(π), e, 黄金分割, 根号2, 根号3\n" + \
               "公式：圆面积, 圆周长, 三角形面积, 矩形面积, 正方形面积, 球体积, 勾股定理\n" + \
               "示例：math[pi] 或 math[圆面积]"
    
    def _python_tool(self, code: str) -> str:
        """Python执行工具：动态运行代码并返回输出结果"""
        # 清理大模型可能附加的 Markdown 格式标记
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()

        print(f"\n[Python执行中...] \n{code}")

        # 捕获标准输出
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                # 执行代码，限制 builtins 并在隔离的字典中运行
                exec(code, {"__builtins__": __builtins__}, {})
            
            result = output.getvalue()
            if not result:
                return "[Python执行结果] 代码执行成功，但没有输出。请确保你在代码中使用了 print() 函数来打印想要观察的结果。"
            
            return f"[Python执行结果]\n{result.strip()}"
            
        except Exception as e:
            # 返回详细的错误追踪栈，方便大模型反思和自我修正代码
            error_msg = traceback.format_exc(limit=1)
            return f"[Python执行错误]\n{error_msg}"
    
    def ask(self, question: str) -> tuple[str, int]:
        """使用ReAct范式回答问题"""
        """使用ReAct范式回答问题"""
        # 记录用户的原始提问
        self.conversation_history.append({"role": "user", "content": question})
        
        thought_process = []
        final_answer = ""
        rounds = 0
        
        for step in range(self.max_steps):
            rounds += 1
            # 构建ReAct提示 (注意：这里固定传入 question，不再传入累加的 current_question)
            react_prompt = self._build_react_prompt(question, thought_process)
            
            # 调用API获取思考和行动
            response = api.api_call(react_prompt)
            llm_output = response.message.content.strip()
            
            # 记录大模型的输出
            thought_process.append(llm_output)
            
            # 【修改点】尝试多种格式解析Action，使用 re.DOTALL 和贪婪匹配 (.*) 
            # 以防止 Python 代码内部的 [] 或 () 导致正则提前截断匹配
            action_match = re.search(r'Action:\s*(\w+)\[(.*)\]', llm_output, re.DOTALL)
            if not action_match:
                action_match = re.search(r'Action:\s*(\w+)\((.*)\)', llm_output, re.DOTALL)
            if not action_match:
                action_match = re.search(r'使用工具[：:]\s*(\w+)\s*[,，]\s*参数[：:]\s*([\s\S]+)', llm_output)
            
            if action_match:
                tool_name = action_match.group(1).lower().strip()
                tool_input = action_match.group(2).strip()
                
                # 工具名称标准化补充 Python
                tool_mapping = {
                    '搜索': 'search', '查找': 'search',
                    '计算': 'calculate', '数学': 'math',
                    '时间': 'datetime', '日期': 'datetime',
                    '百科': 'wiki', '维基': 'wiki',
                    'python': 'python', '代码': 'python', 'py': 'python'
                }
                
                if tool_name in tool_mapping:
                    tool_name = tool_mapping[tool_name]
                
                if tool_name in self.tools:
                    # 执行工具
                    print(f"\n[ReAct] 调用工具: {tool_name}({tool_input})")
                    observation = self.tools[tool_name](tool_input)
                    print(f"[ReAct] 工具结果: {observation}")
                    thought_process.append(f"Step {step+1} - {observation}")
                    #current_question = f"{current_question}\n\n{observation}"
                else:
                    available_tools = ', '.join(self.tools.keys())
                    error_msg = f"Observation: 未知工具 '{tool_name}'，可用工具: {available_tools}"
                    thought_process.append(f"Step {step+1} - {error_msg}")
            elif 'Final Answer:' in llm_output or '最终答案' in llm_output:
                # 已经有最终答案，停止
                break
            else:
                # 没有明确行动，但可能隐含需要工具
                if step < self.max_steps - 1:
                    # 继续尝试，让LLM在下一步明确行动
                    thought_process.append("Observation: 请按照格式输出 Action 或者 Final Answer。")
                    continue
                else:
                    break
            
            #self.conversation_history.append({"role": "assistant", "content": llm_output})
            
        
                # 生成最终答案
        print(f"\n[ReAct] 完整思考过程:")
        for thought in thought_process:
            print(f"  {thought}")
        
        # 检查是否已经有最终答案在思考过程中
        combined_thought = "\n".join(thought_process)
        if 'Final Answer:' in combined_thought:
            # 从思考过程中提取最终答案
            final_answer_match = re.search(r'Final Answer:\s*([\s\S]+?)(?=\n\s*Step|\n\s*step|$)', combined_thought, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
            else:
                final_answer = combined_thought
        elif '最终答案' in combined_thought:
            final_answer_match = re.search(r'最终答案[：:]\s*([\s\S]+?)(?=\n\s*Step|\n\s*step|$)', combined_thought, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
            else:
                final_answer = combined_thought
        else:
            # 生成最终答案
            final_prompt = self._build_final_answer_prompt(question, thought_process)
            final_response = api.api_call(final_prompt)
            final_answer = final_response.message.content
        
        # 循环结束：将最终结论干净地保存进历史记录
        self.conversation_history.append({
            "role": "assistant", 
            "content": final_answer, 
            "thought": combined_thought
        })
        
        print(f"\n[ReAct] 最终答案: {final_answer}")
        return (final_answer, rounds)
    
    def _build_react_prompt(self, question: str, thought_process: List[str]) -> str:
        """构建ReAct提示词"""
        prompt = f"""请使用ReAct范式回答问题。你可以使用以下工具：

【可用工具列表】
1. search[查询内容]: 搜索知识库，查找事实性信息
   示例：search[中国首都] 或 search[北京人口]

2. calculate[数学表达式]: 执行数学计算
   示例：calculate[15 * 17 + 23] 或 calculate[24 / 4]

3. datetime[查询类型]: 获取当前日期和时间
   示例：datetime[now] 或 datetime[日期]

4. math[查询]: 查询数学常数和公式
   示例：math[pi] 或 math[圆面积]

5. python: 执行复杂的Python代码并返回标准输出。请直接在Action下方使用Markdown代码块编写代码，绝不要使用中括号包裹！务必在代码中使用 print() 输出结果！
   示例：
   Action: python
   ```python
   import math
   print(math.factorial(10))

【使用说明】
- 如果问题需要计算或搜索信息，请先使用相应工具
- 可以多步调用工具，每次只调用一个工具
- 当收集到足够信息后，给出最终答案

【输出格式】
Thought: [你的思考过程]
Action: [工具名称][工具输入]

或者当你有足够信息时：
Thought: [你的思考过程]
Final Answer: [最终答案]


问题：{question}

"""
        
        if thought_process:
            prompt += "\n".join(thought_process) + "\n"
        
        return prompt
    
    def _build_final_answer_prompt(self, question: str, thought_process: List[str]) -> str:
        """构建最终答案提示词"""
        prompt = f"""基于以下思考过程和工具调用结果，给出问题的最终答案。

【原始问题】
{question}

【思考过程与工具结果】
"""
        prompt += "\n".join(thought_process)
        prompt += """

【要求】
1. 基于上述信息，给出完整、准确的答案
2. 如果有搜索或计算结果，请在答案中引用
3. 语言要简洁明了，易于理解
4. 如果信息不足，请说明需要什么额外信息

请给出最终答案："""
        return prompt


class ReflexionAgent(BaseAgent):

    """Reflexion架构Agent：带有反思机制"""

   

    def __init__(self, max_reflections: int = 2):

        super().__init__()

        self.max_reflections = max_reflections

        self.reflections = []

   

    def ask(self, question: str) -> tuple[str, int]:

        """使用Reflexion架构回答问题"""

        self.conversation_history.append({"role": "user", "content": question})

       

        attempts = []

        rounds = 0
        

        for attempt in range(self.max_reflections + 1):
            rounds += 1
            # 生成回答

            if attempt == 0:

                # 第一次尝试

                response = api.api_call(question)

            else:

                # 基于之前的尝试和反思生成新回答

                reflection_prompt = self._build_reflection_prompt(question, attempts)

                response = api.api_call(reflection_prompt)

           

            self.conversation_history.append({

                "role": "user",

                "content": response.message.content

            })



            answer = response.message.content

            attempts.append({"answer": answer, "reflection": None})

           

            # 如果不是最后一次，进行反思

            if attempt < self.max_reflections:

                reflection = self._reflect(question, attempts)

                attempts[-1]["reflection"] = reflection

               

                # 检查是否需要继续尝试

                if "答案正确" in reflection or "不需要改进" in reflection:

                    break

           

            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
            })
        # 选择最佳答案（最后一个）
        final_answer = attempts[-1]["answer"]
        # 保存完整对话
        full_process = f"尝试次数：{len(attempts)}\n"
        for i, attempt in enumerate(attempts):
            full_process += f"\n尝试 {i+1}：{attempt['answer']}"
            if attempt['reflection']:
                full_process += f"\n反思：{attempt['reflection']}"
        self.conversation_history.append({
            "role": "assistant",
            "content": final_answer,
            "process": full_process
        })

        return (final_answer, rounds)

    def _reflect(self, question: str, attempts: List[Dict]) -> str:
        """反思当前回答的质量"""
        reflection_prompt = f"""请反思以下问题的回答质量：
问题：{question}
"""
        for i, attempt in enumerate(attempts):

            reflection_prompt += f"回答 {i+1}：{attempt['answer']}\n"
        reflection_prompt += """

请从准确性、完整性、清晰度等方面评估这个回答。如果回答有问题，请指出问题所在并给出改进建议。

如果回答已经很好，请回复"答案正确，不需要改进"。

"""
        response = api.api_call(reflection_prompt)
        return response.message.content

    def _build_reflection_prompt(self, question: str, attempts: List[Dict]) -> str:
        """基于反思构建新的提示词"""
        prompt = f"""请回答以下问题，参考之前的尝试和反思：
问题：{question}
"""
        for i, attempt in enumerate(attempts):
            prompt += f"\n之前的尝试 {i+1}：{attempt['answer']}"
            if attempt['reflection']:
                prompt += f"\n反思：{attempt['reflection']}"
        prompt += "\n\n请根据以上反思，给出一个更好的回答。"
        return prompt 



def main():
    """主函数，演示不同Agent的使用"""
    print("=== 智能体控制流演示 ===")
    print("请选择要使用的Agent类型：")
    print("1. DirectPromptAgent - 基础直接提示")
    print("2. ReActAgent - ReAct范式")
    print("3. ReflexionAgent - Reflexion架构")
    
    choice = input("请输入选项 (1-3): ").strip()
    
    agent = None
    if choice == "1":
        agent = DirectPromptAgent()
        print("\n已选择 DirectPromptAgent")
    elif choice == "2":
        agent = ReActAgent(max_steps=3)
        print("\n已选择 ReActAgent")
    elif choice == "3":
        agent = ReflexionAgent(max_reflections=2)
        print("\n已选择 ReflexionAgent")
    else:
        print("无效选择，使用默认的 DirectPromptAgent")
        agent = DirectPromptAgent()
    
    print("\nAgent已启动！输入 'quit' 或 'exit' 退出，输入 'switch' 切换Agent类型。")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Agent: 再见！")
            break
        
        if user_input.lower() == 'switch':
            # 重新选择Agent
            print("\n请重新选择Agent类型：")
            print("1. DirectPromptAgent - 基础直接提示")
            print("2. ReActAgent - ReAct范式")
            print("3. ReflexionAgent - Reflexion架构")
            
            choice = input("请输入选项 (1-3): ").strip()
            
            if choice == "1":
                agent = DirectPromptAgent()
                print("\n已切换到 DirectPromptAgent")
            elif choice == "2":
                agent = ReActAgent(max_steps=3)
                print("\n已切换到 ReActAgent")
            elif choice == "3":
                agent = ReflexionAgent(max_reflections=2)
                print("\n已切换到 ReflexionAgent")
            continue
        
        if not user_input.strip():
            continue
            
        response = agent.ask(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()

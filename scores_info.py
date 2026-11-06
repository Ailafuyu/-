import json
import math
import re
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
import concurrent.futures


def load_json_docs(file):
    """获得json切片文档"""
    if isinstance(file, str):
        with open(file, "r", encoding="utf-8") as source:
            datas = json.load(source)
    else:
        if hasattr(file, "seek"):
            file.seek(0)
        datas = json.load(file)
    if not isinstance(datas, list):
        raise ValueError("岗位数据必须是 JSON 数组。")
    documents = []
    for index, data in enumerate(datas):
        if not isinstance(data, dict):
            raise ValueError(f"第 {index + 1} 条岗位数据必须是 JSON 对象。")
        documents.append(Document(page_content=json.dumps(data, ensure_ascii=False)))
    return documents


def get_retriever(texts,db_key,db_url,db_model):
    """存储/检索器"""
    embedding_model = OpenAIEmbeddings(
        model=db_model,
        api_key=SecretStr(db_key),
        base_url=db_url,
        timeout=120
    )
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 50,
        }
    )
    return retriever


@lru_cache(maxsize=8)
def get_model(key,url,model,temperature):
    """获得语言大模型"""
    llm = ChatOpenAI(
        model=model,
        api_key=SecretStr(key),
        base_url=url,
        temperature=temperature,
    )
    return llm


def get_job_score(key, url, model,job_info,json_profile):
    llm = get_model(key, url, model, 0)
    system_text = """
    你是一名资深职业规划专家、招聘专家和人才匹配算法专家。
    # 任务
    根据【用户JSON画像】和【当前岗位信息】，客观评估该岗位与候选人的匹配程度，并输出评分结果。
    你的职责仅限于对当前岗位进行独立评估。
    不要与其他岗位比较。
    不要考虑排名。
    不要推荐其他岗位。
    不要补充任何岗位信息或候选人信息。
    所有分析必须严格依据输入数据。
    ==================================================

    # 输入
    ## 用户职业画像
    由用户输入，包含：基本信息、教育背景、工作经历、项目经历、技能能力、专业能力、通用能力、职业目标、发展方向、求职偏好
    ==================================================

    # 当前岗位
    岗位信息：
    {job_info}
    ==================================================

    # 分析流程（必须严格执行）
    ## Step1：理解候选人
    分析候选人的：教育背景、工作经历、项目经历、行业经验、专业技能、工具技能、通用能力、管理能力、沟通能力、数据分析能力、项目推进能力、可迁移能力、职业目标、长期发展方向
    重点识别：候选人真正能够创造价值的能力，而不是仅根据职位名称或技能关键词进行判断。
    例如：
    销售经验可以迁移为：客户沟通、商务谈判、需求分析
    项目经验可以迁移为：项目管理、跨部门协作、推进能力
    运营经验可以迁移为：数据分析、用户增长、活动策划
    ==================================================
    ## Step2：理解岗位
    分析岗位：核心职责、日常工作、必备能力、加分能力、工具要求、行业要求、学历要求、工作经验要求、岗位发展空间
    不要仅依据岗位名称判断岗位。
    应依据岗位职责进行分析。
    ==================================================
    ## Step3：能力匹配分析
    判断：
    候选人的能力属于：完全匹配、部分匹配、潜力匹配、不匹配
    重点考虑：已有能力、可迁移能力、学习成本、能否快速胜任
    ==================================================
    ## Step4：职业发展分析
    分析：
    岗位是否能够：延续已有经验、提升核心竞争力、符合职业目标、提供成长空间、提高未来职业价值
    若岗位属于明显降级岗位，应降低评分。
    ==================================================
    ## Step5：硬性条件分析
    重点检查：学历、专业、工作年限、行业经验、必要证书、特殊要求
    若存在明显无法满足的要求，应降低评分。
    ==================================================

    # 评分规则（总分100）
    采用减分原则进行评分。
    初始分数为100分。
    根据能力缺口、经验缺口、职业方向偏离、硬性条件限制、岗位发展价值等因素逐项扣分。
    最终得到综合评分。
    ==================================================
    ## 1. 能力匹配（ability_match，满分35分）
    评价：专业技能、工具技能、行业经验、通用能力、可迁移能力
    取值范围：0 <= ability_match <= 35
    ==================================================
    ## 2. 工作职责匹配（responsibility_match，满分25分）
    评价：候选人过去经历是否能够支撑岗位主要职责。
    取值范围：0 <= responsibility_match <= 25
    ==================================================
    ## 3. 职业方向匹配（career_match，满分20分）
    评价：
    岗位是否符合：长期职业规划、职业目标、未来发展方向
    取值范围：0 <= career_match <= 20
    ==================================================
    ## 4. 背景条件匹配（background_match，满分10分）
    评价：学历、专业、工作经验、行业背景
    取值范围：0 <= background_match <= 10
    ==================================================
    ## 5. 岗位发展价值（development_value，满分10分）
    评价：
    岗位是否能够：提升能力、增加职业竞争力、提供成长空间、提升长期职业价值
    取值范围：0 <= development_value <= 10
    ==================================================
    综合得分计算方式：
    match_score =ability_match +responsibility_match +career_match +background_match +development_value
    禁止根据整体印象调整总分。总分只能由五项相加得到。不得在计算完成后再次上调或下调总分。
    ==================================================
    # 评分参考
    95~100：高度匹配，可直接胜任，且符合长期发展。
    90~94：匹配度很高，仅存在少量不足。
    80~89：整体匹配，需要少量学习即可胜任。
    70~79：存在一定能力差距，但具有较好的可迁移能力。
    60~69：存在明显能力不足，仅作为备选。
    60以下：匹配度较低。
    ==================================================
    # 降分规则
    出现以下情况，应明显降低评分：
    - 岗位主要职责与候选人经历关联较弱
    - 职业方向明显不一致
    - 无法满足关键硬性要求
    - 属于明显降级岗位
    - 可迁移能力较弱
    - 成长空间有限

    不得因为：
    - 岗位名称相似
    - 技能关键词相同
    - 使用相同工具
    而直接提高评分。

    ==================================================

    # 输出要求
    所有结论必须严格依据：用户JSON画像、当前岗位信息
    不得编造任何经历、技能、岗位要求或行业背景。
    推荐理由必须具体、可验证，不得使用空泛描述。
    matching_strengths 仅列举真正匹配岗位的核心优势。
    possible_gaps 仅列举真正影响录用的能力差距，不得使用“经验不足”“能力待提升”等空泛描述。
    ==================================================
    # 输出格式
    仅输出 JSON，不输出任何解释。
    {{
      "match_score": 0,
      "ability_match": 0,
      "responsibility_match": 0,
      "career_match": 0,
      "background_match": 0,
      "development_value": 0,
      "recommend_reason": "",
      "matching_strengths": [
        ""
      ],
      "possible_gaps": [
        ""
      ],
      "development_analysis": ""
    }}
    ==================================================
    在输出 JSON 前，必须完成全部分析过程，但不要输出分析过程，仅输出最终 JSON。
    """
    human_text = """
    这是我的json画像：
    {user_profile}
    请据此给岗位打分。
    """

    prompt_Template = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", human_text)
        ]
    )

    chain = prompt_Template | llm
    response = chain.invoke({"user_profile": json_profile, "job_info": job_info})
    return response.content



_SCORE_LIMITS = {
    "ability_match": 35,
    "responsibility_match": 25,
    "career_match": 20,
    "background_match": 10,
    "development_value": 10,
}


def _parse_model_json(response):
    """Parse a JSON object returned by the model, tolerating Markdown fences."""
    if not isinstance(response, str):
        raise ValueError("模型返回内容不是文本。")
    cleaned = response.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型没有返回有效的 JSON 对象。")
        result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("模型返回的 JSON 顶层必须是对象。")
    return result


def _normalize_score_result(result):
    """Validate score fields, clamp them to their rubric, and recompute total."""
    normalized = dict(result)
    for field, maximum in _SCORE_LIMITS.items():
        value = normalized.get(field)
        if isinstance(value, bool):
            raise ValueError(f"评分字段 {field} 必须是数字。")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"评分字段 {field} 必须是数字。")
        if not math.isfinite(value):
            raise ValueError(f"评分字段 {field} 必须是有限数字。")
        normalized[field] = min(max(value, 0.0), float(maximum))
    normalized["match_score"] = round(sum(normalized[field] for field in _SCORE_LIMITS), 2)
    return normalized


def process_single_job1(task):
    """单条岗位的处理逻辑（带二次重试机制），作为线程池的执行任务"""
    index, job_info, key, url, model, json_profile=task
    numb = index + 1
    print(f"正在给第 {numb} 条岗位打分")
    attempt_max=3
    for attempt in range(1, attempt_max + 1):
        try:
            if not isinstance(job_info, dict):
                raise ValueError("岗位数据必须是 JSON 对象。")
            response = get_job_score(key, url, model, job_info, json_profile)
            result = _normalize_score_result(_parse_model_json(response))
            return job_info | result
        except Exception:
            if attempt < attempt_max:
                print(f"⚠️ 第 {numb} 条岗位第 {attempt} 次处理失败，正在重试...")
            else:
                print(f"❌ 第 {numb} 条岗位经过 {attempt_max} 次尝试均失败，已跳过")
                return None


def process_single_job2(task):
    """单条岗位的处理逻辑（带二次重试机制），作为线程池的执行任务"""
    index, job_info, key, url, model, json_profile=task
    job_info = json.loads(job_info.page_content)
    if not isinstance(job_info, dict):
        raise ValueError("检索到的岗位数据必须是 JSON 对象。")
    numb = index + 1
    print(f"正在给第 {numb} 条岗位打分")
    attempt_max=3
    for attempt in range(1, attempt_max + 1):
        try:
            response = get_job_score(key, url, model, job_info, json_profile)
            result = _normalize_score_result(_parse_model_json(response))
            return job_info | result
        except Exception:
            if attempt < attempt_max:
                print(f"⚠️ 第 {numb} 条岗位第 {attempt} 次处理失败，正在重试...")
            else:
                print(f"❌ 第 {numb} 条岗位经过 {attempt_max} 次尝试均失败，已跳过")
                return None

def save_to_json(jobs_info):
    """json格式保存"""
    with open('评分数据gpt1.json', 'w', encoding="utf-8") as f:
        json.dump(jobs_info, f, ensure_ascii=False, indent=2)


def scores_total(uploaded_file, json_profile, key, url, model,log):
    if isinstance(uploaded_file, str):  # 文件路径
        with open(uploaded_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
    else:
        docs = json.load(uploaded_file)
    tasks = [(index, job_info, key, url, model, json_profile) for index, job_info in enumerate(docs)]
    results = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(tasks)))) as executor:
        futures = {executor.submit(process_single_job1, task): task[0] for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results[futures[future]] = future.result()
            log(f"已分析完{completed}条岗位")
    new_job_infos = [item for item in results if item is not None]
    log(f"分析已完成，共分析{len(new_job_infos)}条岗位信息")
    return new_job_infos


def scores_rag(db_key,db_url,db_model,uploaded_file,json_profile,key,url,model,log):
    docs=load_json_docs(uploaded_file)
    retriever=get_retriever(docs,db_key,db_url,db_model)
    retrieved_docs=retriever.invoke(json_profile)
    tasks = [(index, job_info, key, url, model, json_profile) for index, job_info in enumerate(retrieved_docs)]
    results = [None] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(tasks)))) as executor:
        futures = {executor.submit(process_single_job2, task): task[0] for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results[futures[future]] = future.result()
            log(f"已分析完{completed}条岗位")
    new_job_infos = [item for item in results if item is not None]
    log(f"分析已完成，共分析{len(new_job_infos)}条岗位信息")
    return new_job_infos





from ddgs import DDGS
print("导入成功")
with DDGS() as ddgs:
    res = ddgs.text(
        "人工智能",
        region="cn-zh",
        safesearch="on",
        max_results=5
    )
    print(list(res))
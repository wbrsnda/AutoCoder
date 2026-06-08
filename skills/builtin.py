"""内置技能定义。"""

from typing import Optional

SKILLS = {
    "explore_workspace": {
        "description": "列出工作目录文件并总结内容",
        "triggers": [
            "有什么文件", "写了什么", "目录下有什么", "告诉我当前工作目录",
            "what files", "what's in", "list files", "show me",
            "工作目录", "workspace", "项目里有什么", "看看文件"
        ],
        "steps": [
            "1. DELEGATE TO CODER: list_dir to see what files exist",
            "2. After getting the file list, DELEGATE TO CODER: read_file for each file",
            "3. After reading all files, summarize what each file does",
        ]
    },

    "delete_file": {
        "description": "删除工作目录下的单个指定文件",
        "triggers": [
            "删除", "delete", "remove", "删掉", "删除文件"
        ],
        "steps": [
            "1. 先告诉用户将要删除哪个文件，并明确询问：'是否确认删除该文件？请回复 YES 或 NO'。",
            "2. 只有当用户明确回复 YES 后，才调用 mcp_delete_file 工具进行删除。",
            "3. 删除完成后，REPORT TO ARCHITECT。"
        ]
    },

    "delete_workspace": {
        "description": "安全删除工作目录下所有文件",
        "triggers": [
            "清空", "删除全部", "remove all", "clean workspace"
        ],
        "steps": [
            "1. DELEGATE TO CODER: list_dir to see what files exist",
            "2. Tell the user exactly which files will be deleted",
            "3. Ask: 'These files will be permanently deleted. Reply YES to confirm.' Then output AWAITING USER INPUT",
        ]
    },
}


def match_skill(user_message: str) -> Optional[dict]:
    """检查用户消息是否匹配某个内置技能。"""
    msg_lower = user_message.lower()
    for name, skill in SKILLS.items():
        for trigger in skill["triggers"]:
            if trigger in msg_lower:
                return skill
    return None
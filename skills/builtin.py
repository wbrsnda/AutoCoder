"""内置技能定义。"""

from typing import Optional

SKILLS = {
    "explore_workspace": {
        "description": "列出工作目录文件并总结目录内容",
        "triggers": [
            "有什么文件", "写了什么", "目录下有什么", "告诉我当前工作目录",
            "what files", "what's in", "list files", "show me",
            "工作目录", "workspace", "项目里有什么", "看看文件"
        ],
        "steps": [
            "1. DELEGATE TO CODER: Use mcp_list_dir to list the workspace root.",
            "2. Summarize the directory contents for the user.",
            "3. ONLY read specific files if the user explicitly asks for those files.",
        ]
    },

    "delete_file": {
        "description": "删除工作目录下的单个指定文件",
        "triggers": [
            "删除", "delete", "remove", "删掉", "删除文件"
        ],
        "steps": [
            "1. Tell the user exactly which file will be deleted.",
            "2. Ask for confirmation: '是否确认删除该文件？请回复 YES 或 NO' and end with AWAITING USER INPUT.",
            "3. Only after the user explicitly replies YES, DELEGATE TO CODER to call mcp_delete_file.",
            "4. After deletion completes, answer the user and end with AWAITING USER INPUT."
        ]
    },

    "delete_workspace": {
        "description": "安全删除工作目录下所有文件",
        "triggers": [
            "清空", "删除全部", "remove all", "clean workspace"
        ],
        "steps": [
            "1. DELEGATE TO CODER: Use mcp_list_dir to list the workspace root.",
            "2. Tell the user exactly which files will be deleted.",
            "3. Ask: 'These files will be permanently deleted. Reply YES to confirm.' Then output AWAITING USER INPUT.",
        ]
    },
}


def match_skill(user_message: str) -> Optional[dict]:
    """检查用户消息是否匹配某个内置技能。"""
    msg_lower = user_message.lower()
    for _, skill in SKILLS.items():
        for trigger in skill["triggers"]:
            if trigger in msg_lower:
                return skill
    return None
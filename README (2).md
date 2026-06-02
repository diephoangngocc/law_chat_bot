## Cách dùng

Dừng server:

```powershell
Ctrl + C
```

Backup:

```powershell
Copy-Item .\backend .\backend_backup_context_fix -Recurse -Force
```

Giải nén patch vào repo:

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\law_chatbot_v2_smalltalk_llm_context_fix.zip" -DestinationPath "D:\law_chatbot_v2" -Force
```

Chạy lại:

```powershell
python run.py
```

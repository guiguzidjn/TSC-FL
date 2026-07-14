import requests, base64, os, json

TOKEN = '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'
USER = 'guiguzidjn'
REPO = 'TSC-FL'
HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github+json',
}

# Step 1: Create private repo
print('Creating private repo...')
r = requests.post('https://api.github.com/user/repos', json={
    'name': REPO,
    'private': True,
    'description': 'Two-Stage Classifier for Federated Learning (TSC-FL) - IoT Intrusion Detection',
}, headers=HEADERS)
if r.status_code == 201:
    print(f'Repo created: {r.json()["html_url"]}')
elif r.status_code == 422:
    print('Repo already exists, using existing')
else:
    print(f'Error: {r.status_code} {r.text[:300]}')
    exit(1)

# Step 2: Collect files
PROJECT_ROOT = r'D:\pythonProject\TSC-FL'
EXCLUDE_DIRS = {'.venv', '__pycache__', '.idea', 'multi_model', 'confusion_matrix'}
EXCLUDE_EXTS = {'.pth', '.pt'}
EXCLUDE_FILES = {'temp_github.py'}

files_to_upload = []
for root, dirs, files in os.walk(PROJECT_ROOT):
    # Skip excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext in EXCLUDE_EXTS or f in EXCLUDE_FILES:
            continue
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace('\\', '/')
        # Skip files in excluded subdirs
        skip = False
        for ed in EXCLUDE_DIRS:
            if ed in rel_path.split('/'):
                skip = True
                break
        if not skip:
            files_to_upload.append((full_path, rel_path))

print(f'Files to upload: {len(files_to_upload)}')
for _, rp in sorted(files_to_upload):
    print(f'  {rp}')

# Step 3: Upload files
success = 0
failed = 0
for full_path, rel_path in files_to_upload:
    with open(full_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode()
    
    url = f'https://api.github.com/repos/{USER}/{REPO}/contents/{rel_path}'
    r = requests.put(url, json={
        'message': f'Add {rel_path}',
        'content': content,
    }, headers=HEADERS)
    
    if r.status_code in (201, 200):
        success += 1
        print(f'  OK: {rel_path}')
    elif r.status_code == 422 and 'already exists' in r.text:
        # Update existing
        # Need to get SHA first
        r_get = requests.get(url, headers=HEADERS)
        if r_get.status_code == 200:
            sha = r_get.json()['sha']
            r_update = requests.put(url, json={
                'message': f'Update {rel_path}',
                'content': content,
                'sha': sha,
            }, headers=HEADERS)
            if r_update.status_code in (201, 200):
                success += 1
                print(f'  Updated: {rel_path}')
            else:
                failed += 1
                print(f'  FAIL update: {rel_path} {r_update.status_code}')
        else:
            failed += 1
            print(f'  FAIL get SHA: {rel_path}')
    else:
        failed += 1
        print(f'  FAIL: {rel_path} {r.status_code} {r.text[:200]}')

print(f'\nDone! Success: {success}, Failed: {failed}')
print(f'Repo URL: https://github.com/{USER}/{REPO}')

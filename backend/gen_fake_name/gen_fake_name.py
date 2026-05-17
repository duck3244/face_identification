from faker import Faker
import json

# Faker 인스턴스 생성 (영어 이름용)
fake = Faker('en_US')  # 영어(미국) 로케일 사용

# 10개의 이름을 담을 리스트 생성
names = []

# 10개의 이름 생성
for _ in range(10):
    name_data = {
        "full_name": fake.name(),
        "first_name": fake.first_name(),
        "last_name": fake.last_name()
    }
    names.append(name_data)

# JSON 파일로 저장
with open('fake_names.json', 'w', encoding='utf-8') as json_file:
    json.dump(names, json_file, indent=4)

print("fake_names.json 파일이 생성되었습니다.")
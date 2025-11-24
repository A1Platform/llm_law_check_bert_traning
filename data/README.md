# 학습 데이터셋

## 디렉터리 구성

```
datasets/
├── train/
│   ├── trainset_cosmetics.csv         # 화장품 광고 문구 학습 데이터 (9,428 건)
│   └── trainset_health_food.csv       # 건강기능식품 광고 문구 학습 데이터 (396 건)
├── test/
│   ├── testset_cosmetics_line.csv     # 문장 단위 화장품 검증 세트 (188 건)
│   └── testset_landing_page.csv       # 랜딩페이지 URL 검증 세트 (50 건)
└── README.md
```

## 파일 상세

| 파일 | 주요 컬럼 | 설명 |
| --- | --- | --- |
| `train/trainset_cosmetics.csv` | `text`, `label`, `tag` | 화장품 광고 문구. `label`은 규제 위반 여부(`OK`, `NG`), `tag`는 데이터 출처(예: `라인_반려문구`, `약기법_솔루션_로그`). |
| `train/trainset_health_food.csv` | `text`, `label`, `tag` | 건강기능식품 관련 광고 문구. 동일한 라벨/태그 체계를 사용 |
| `test/testset_cosmetics_line.csv` | `text`, `label` | 학습 데이터와 동일한 형식의 화장품 광고 문장 단위 검증 세트. 모델 추론 후 평가에 활용 |
| `test/testset_landing_page.csv` | `url`, `label` | Qoo10의 랜딩페이지 URL을 기반으로 한 검증 세트. URL 단위로 규제 위반 여부를 판정한다. |

## 주의 사항

- **학습/검증 분리**: `train` 디렉터리는 파인튜닝에만 사용하고, `test` 디렉터리는 모델 최종 검증용으로 유지합니다.  
또한 데이터 출처(`tag`) 구분을 위해 학습 데이터 세트에 검증 세트의 데이터가 포함되어 있을 수 있으므로, 모델 학습시 필히 검증 데이터 세트를 제외하는 전처리 과정이 필요합니다.
- **랜딩페이지 관련**: 랜딩 페이지 검증 데이터의 경우, 해당 페이지의 상품 상세 이미지 추출 및 OCR 전처리 작업을 필요로 합니다.

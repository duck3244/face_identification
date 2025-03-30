import faiss
import pickle
import numpy as np

from face_representation import represent_face


class FaceDatabase:
    def __init__(self, model_name, distance_metric):
        """
        얼굴 데이터베이스 초기화

        Parameters:
        -----------
        model_name: str
            얼굴 표현을 추출하는 데 사용할 모델 이름
        distance_metric: str
            유사성 측정에 사용할 거리 측정 방법 (cosine, euclidean, euclidean_l2)
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.embeddings = []
        self.identities = []
        self.index = None


    def add_face(self, img_path, identity):
        """
        얼굴을 데이터베이스에 추가

        Parameters:
        -----------
        img_path: str
            데이터베이스에 추가할 얼굴 이미지 경로
        identity: str
            얼굴 이미지에 연결할 신원 정보

        Returns:
        --------
        bool
            성공 여부
        """
        embedding = represent_face(img_path, self.model_name)

        if embedding is not None:
            self.embeddings.append(embedding)
            self.identities.append(identity)
            # 데이터베이스가 업데이트되었으므로 인덱스 재구축
            self._build_index()
            return True
        else:
            return False


    def _build_index(self):
        """FAISS 인덱스 구축"""
        if len(self.embeddings) == 0:
            print("데이터베이스가 비어 있습니다.")
            return

        # 임베딩을 numpy 배열로 변환
        embeddings_array = np.array(self.embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        # FAISS 인덱스 생성
        if self.distance_metric == "cosine":
            # 코사인 유사도를 위한 L2 정규화 및 내적 인덱스
            faiss.normalize_L2(embeddings_array)
            self.index = faiss.IndexFlatIP(dimension)  # 내적 제품 (코사인 유사도)
        else:
            # 유클리드 거리를 위한 L2 인덱스
            self.index = faiss.IndexFlatL2(dimension)

        # 인덱스에 벡터 추가
        self.index.add(embeddings_array)
        print(f"FAISS 인덱스가 {len(self.embeddings)}개의 얼굴로 구축되었습니다.")


    def search(self, query_embedding, threshold, top_k):
        """
        데이터베이스에서 유사한 얼굴 검색

        Parameters:
        -----------
        query_embedding: numpy.ndarray
            검색할 쿼리 임베딩
        threshold: float
            얼굴 일치로 간주할 임계값
        top_k: int
            반환할 상위 일치 수

        Returns:
        --------
        list
            [(identity, similarity_score), ...] 형식의 일치 목록
        """
        if self.index is None or len(self.embeddings) == 0:
            print("데이터베이스가 비어 있습니다.")
            return []

        # 쿼리 임베딩을 numpy 배열로 변환
        query_embedding = np.array([query_embedding]).astype('float32')

        # 코사인 유사도의 경우 L2 정규화
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query_embedding)

        # FAISS를 사용하여 가장 가까운 이웃 검색
        distances, indices = self.index.search(query_embedding, top_k)

        # 결과 형식화
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = distances[0][i]

            if self.distance_metric == "cosine":
                # 코사인 유사도의 경우 점수가 높을수록 더 유사함
                if score >= threshold:
                    results.append((self.identities[idx], float(score)))
            else:
                # 유클리드 거리의 경우 점수가 낮을수록 더 유사함
                # 유클리드 거리를 [0, 1] 범위의 유사도 점수로 변환
                similarity = 1 / (1 + score)
                if similarity >= threshold:
                    results.append((self.identities[idx], float(similarity)))

        return results


    def save(self, file_path):
        """데이터베이스를 파일에 저장"""
        data = {
            'embeddings': self.embeddings,
            'identities': self.identities,
            'model_name': self.model_name,
            'distance_metric': self.distance_metric
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"데이터베이스가 {file_path}에 저장되었습니다.")


    def load(self, file_path):
        """파일에서 데이터베이스 로드"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.embeddings = data['embeddings']
            self.identities = data['identities']
            self.model_name = data['model_name']
            self.distance_metric = data['distance_metric']

            # 로드 후 인덱스 재구축
            self._build_index()
            print(f"데이터베이스가 {file_path}에서 로드되었습니다.")
            return True
        except Exception as e:
            print(f"데이터베이스 로드 중 오류 발생: {e}")
            return False


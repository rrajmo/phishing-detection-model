import joblib
import lightgbm as lgb
from typing import List, Any, Optional
from models.base_model import BaseModel

class LightGBMClassifier(BaseModel):
    def __init__(self, parameters: Optional[List[str]] = None) -> None:
        self.model = lgb.LGBMClassifier(**(parameters or {}))
        self.parameters = parameters

    def fit(self, X: Any, y: Any, categorical_feature: Any = None) -> lgb.LGBMClassifier:
        return self.model.fit(X, y, categorical_feature=categorical_feature)

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)
    
    def predict_probabilities(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def save(self, file: str) -> None:
        joblib.dump(self.model, file)

    @classmethod
    def load(self, file: str) -> lgb.LGBMClassifier:
        self.model = joblib.load(file)
        return self.model

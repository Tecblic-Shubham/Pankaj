from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import train


models = {
'et': ExtraTreesClassifier( criterion = 'entropy',
                            class_weight='balanced', 
                            random_state=42, 
                            n_estimators=1000,
                            warm_start=True, 
                            max_samples=None, 
                            bootstrap=True,
                            max_depth=12,
                            ),
'r_f': RandomForestClassifier(
                            max_depth=12,
                            class_weight="balanced_subsample",
                            warm_start=True
)

}
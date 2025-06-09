# Compara dois modelos Naive Bayes (GaussianNB e BernoulliNB) e os avalia usando Acurácia e F1-Score

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

breastCancer = load_breast_cancer()
iris = load_iris()

breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test = train_test_split(
    breastCancer.data, breastCancer.target, test_size=0.2, random_state=42)

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

gaussian_nb_classifier = GaussianNB()
bernoulli_nb_classifier = BernoulliNB()

gaussian_nb_classifier.fit(breast_cancer_X_train, breast_cancer_y_train)
bernoulli_nb_classifier.fit(breast_cancer_X_train, breast_cancer_y_train)

gaussian_nb_pred = gaussian_nb_classifier.predict(breast_cancer_X_test)
bernoulli_nb_pred = bernoulli_nb_classifier.predict(breast_cancer_X_test)

gaussian_nb_accuracy = accuracy_score(breast_cancer_y_test, gaussian_nb_pred)
gaussian_nb_f1 = f1_score(breast_cancer_y_test, gaussian_nb_pred)

bernoulli_nb_accuracy = accuracy_score(breast_cancer_y_test, bernoulli_nb_pred)
bernoulli_nb_f1 = f1_score(breast_cancer_y_test, bernoulli_nb_pred)

print(f'Acurácia gaussiana: {gaussian_nb_accuracy}\nF1-score gaussiano: {gaussian_nb_f1}')
print(f'\nAcurácia bernoulli: {bernoulli_nb_accuracy}\nF1-score bernoulli: {bernoulli_nb_f1}')

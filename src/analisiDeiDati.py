import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from imblearn.over_sampling import SMOTE
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


def format():
    file_path = 'UsedCombined.txt'

    data = pd.read_csv(file_path, sep='\t')
    data = data.drop(columns=['Unnamed: 0'])

    output_path = 'UsedCombined2.csv'
    data.to_csv(output_path, index=False)

def dimensionalita(data):
    dimensionalita = data.shape

    print(dimensionalita)


def descr(data):
    print(data.info())


def isNull(data):
    null_counts = data.isnull().sum()
    print(null_counts)


def duplicati(data):
    duplicati = data.duplicated()
    numero_duplicati = duplicati.sum()

    print(numero_duplicati)


def rimuovi_duplicati(data):
    data_senza_duplicati = data.drop_duplicates()

    return data_senza_duplicati

def clean_coloumns(data):
    data_cleaned = data.drop(columns=['neutrophil'])
    data_cleaned = data_cleaned.drop(columns=['lymphocytes'])
    data_cleaned = data_cleaned.drop(columns=['serumLevelsOfWhiteBloodCell'])
    data_cleaned = data_cleaned.drop(columns=['Diagnosis'])

    return data_cleaned

def impute_media(data, colonna):

    media = data[colonna].mean()
    data[colonna].fillna(media, inplace=True)

    return data

def impute_moda(data, colonna):

    moda = data[colonna].mode()

    if not moda.empty:
        data[colonna].fillna(moda[0], inplace=True)

    return data

def impute_temperatura(data):
    
    data.loc[(data['Fever'] == 'Yes') & (data['Temperature'].isnull()), 'Temperature'] = 38.0
    data.loc[(data['Fever'] == 'No') & (data['Temperature'].isnull()), 'Temperature'] = 36.5


def modifica_valori_fever(data):
    data.loc[data['Temperature'] >= 37.0, 'Fever'] = 'Yes'
    data.loc[data['Temperature'] < 37.0, 'Fever'] = 'No'

def imputa_x_ray(data):

    data["XrayResults"].fillna("Non_Presente", inplace=True)

    return data

def imputa_CTscanResults(data):

    data["CTscanResults"].fillna("Non_Presente", inplace=True)

    return data

def dummy_value(data):
    data = pd.get_dummies(data, dtype='int')

    return data

def normalizzation_temp(data):
    colonna_da_normalizzare = data['Temperature']

    scaler = MinMaxScaler()

    data['Temperature'] = scaler.fit_transform(colonna_da_normalizzare.values.reshape(-1, 1))

    return data

def normalizzation_age(data):
    colonna_da_normalizzare = data['Age']

    scaler = MinMaxScaler()

    data['Age'] = scaler.fit_transform(colonna_da_normalizzare.values.reshape(-1, 1))

    return data

def show(data):

    class_counts = data['D'].value_counts()

    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title('H1N1 vs COVID19')
    plt.xlabel('Numero classi')
    plt.ylabel('Numero di record')
    plt.xticks(rotation=0)
    plt.show()

def distr(data):
    counts = data.groupby(['D', 'Sex_F']).size().unstack()

    print(counts)
    labels = {0: 'H1N1', 1: 'COVID19'}
    counts.index = counts.index.map(labels)

    counts.plot(kind='bar', stacked=True)
    plt.xlabel('Sesso')
    plt.ylabel('Numero di casi totale')
    plt.title('Distribuzione della variabile dipendente rispetto al sesso')
    plt.xticks(rotation=0)
    plt.legend(title='Genere', labels=['Femminile', 'Maschile'])
    plt.show()

def distr_Age(data):

    sns.histplot(data, x='Age', hue='D', element='step', stat='density', common_norm=False,
                 palette={0: 'blue', 1: 'red'})
    plt.xlabel('Età')
    plt.ylabel('Densità')
    plt.title('Distribuzione dell\'età per malati di Covid e influenza')
    plt.legend(title='Malattia', labels=['Covid', 'Influenza'])
    plt.show()

def distr_temp(data):
    sns.histplot(data, x='Temperature', hue='D', element='step', stat='density', common_norm=False,
                 palette={0: 'blue', 1: 'red'})
    plt.xlabel('Temperatura')
    plt.ylabel('Densità')
    plt.title('Distribuzione della temperatura per malati di Covid e influenza')
    plt.legend(title='Malattia', labels=['Covid', 'Influenza'])
    plt.show()

def distr_corr(data):
    plt.figure(figsize=(13.6, 13.6))
    heat_map = plt.matshow(data.corr(), cmap='coolwarm', fignum=0)
    plt.colorbar(heat_map).ax.tick_params(labelsize=8)

    plt.xticks(ticks=np.arange(data.shape[1]), labels=data.columns, fontsize=8, rotation=90)
    plt.yticks(ticks=np.arange(data.shape[1]), labels=data.columns, fontsize=8)

    plt.title('Matrice di Correlazione', pad=10)
    plt.subplots_adjust(left=0.3, bottom=0.1)
    plt.show()

def decision_tree_impl(data):

    data = normalizzation_temp(data)
    data = normalizzation_age(data)

    X = data.drop('D', axis=1)
    y = data['D']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    undersampler = RandomUnderSampler(random_state=101)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    smote = SMOTE(random_state=101)
    X_train, y_train = smote.fit_resample(X_train_under, y_train_under)

    model = DecisionTreeClassifier(
        criterion="gini", random_state=100,
        max_depth=9, min_samples_leaf=5)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(30, 25))
    plot_tree(model, filled=True)
    plt.show()

    feature_importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(13.5, 10))
    plt.title("Feature Importances")

    # Grafico a barre orizzontali
    plt.barh(range(X.shape[1]), feature_importances[indices], align="center")
    plt.yticks(range(X.shape[1]), features[indices], fontsize=8)
    plt.ylim([-1, X.shape[1]])
    plt.gca().invert_yaxis()  # Inverti l'asse y per avere le feature più importanti in cima

    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature Names")

    plt.subplots_adjust(left=0.3)

    plt.show()

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap_values_mean = np.abs(shap_values.values).mean(axis=(0, 2))

    feature_names = X_test.columns

    indices = np.argsort(shap_values_mean)[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), shap_values_mean[indices], align="center")
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=8)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Feature Importance based on SHAP Values")
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.4)
    plt.show()


def decision_tree_valutation_depth(data):
    
    data = normalizzation_temp(data)
    data = normalizzation_age(data)

    X = data.drop('D', axis=1)
    y = data['D']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    undersampler = RandomUnderSampler(random_state=101)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    smote = SMOTE(random_state=101)
    X_train, y_train = smote.fit_resample(X_train_under, y_train_under)

    train_errors = []
    val_errors = []
    depths = range(1, 50)

    for depth in depths:
        model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=depth, min_samples_leaf=5)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        train_errors.append(train_error)

        y_val_pred = model.predict(X_test)
        val_error = 1 - accuracy_score(y_test, y_val_pred)
        val_errors.append(val_error)

    plt.figure(figsize=(10, 6))

    plt.plot(depths, train_errors, label='Training Error', marker='o')
    plt.plot(depths, val_errors, label='Validation Error', marker='o')

    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.title('Error vs. Tree Depth')
    plt.legend()
    plt.grid(True)

    plt.show()


def random_forest_valutation_depth(data):
    data = normalizzation_temp(data)
    data = normalizzation_age(data)

    X = data.drop('D', axis=1)
    y = data['D']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    undersampler = RandomUnderSampler(random_state=101)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    smote = SMOTE(random_state=101)
    X_train, y_train = smote.fit_resample(X_train_under, y_train_under)

    train_errors = []
    val_errors = []
    depths = range(1, 50)

    for depth in depths:
        model = RandomForestClassifier(max_depth= depth, random_state=100).fit(X_train, y_train)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        train_errors.append(train_error)

        y_val_pred = model.predict(X_test)
        val_error = 1 - accuracy_score(y_test, y_val_pred)
        val_errors.append(val_error)

    plt.figure(figsize=(10, 6))

    plt.plot(depths, train_errors, label='Training Error', marker='o')
    plt.plot(depths, val_errors, label='Validation Error', marker='o')

    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.title('Error vs. Tree Depth')
    plt.legend()
    plt.grid(True)

    plt.show()

def rndmforrest(data):

    X = data.drop('D', axis=1)
    y = data['D']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    undersampler = RandomUnderSampler(random_state=101)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    smote = SMOTE(random_state=101)
    X_train, y_train = smote.fit_resample(X_train_under, y_train_under)

    model = RandomForestClassifier(max_depth=13, random_state=100).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    feature_importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(13.5, 10))
    plt.title("Feature Importances in Random Forest")

    plt.barh(range(X.shape[1]), feature_importances[indices], align="center")
    plt.yticks(range(X.shape[1]), features[indices], fontsize=8)
    plt.ylim([-1, X.shape[1]])
    plt.gca().invert_yaxis()  # Inverti l'asse y per avere le feature più importanti in cima

    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature Names")

    plt.subplots_adjust(left=0.3)

    plt.show()

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)

    shap_values_mean = np.abs(shap_values.values).mean(axis=(0, 2))
    feature_names = X_test.columns
    indices = np.argsort(shap_values_mean)[::-1]

    plt.figure(figsize=(13.5, 10))
    plt.title("Feature Importances based on SHAP Values")

    plt.barh(range(len(indices)), shap_values_mean[indices], align="center")
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=8)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature Names")
    plt.gca().invert_yaxis()

    plt.subplots_adjust(left=0.3)
    plt.show()

def __main__():
    file_path = 'UsedCombined.csv'
    data = pd.read_csv(file_path)

    decision_tree_impl(data)
    

if __name__ == '__main__':
    __main__()
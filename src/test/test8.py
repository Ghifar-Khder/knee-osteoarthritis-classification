import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# --------------------------------------------------
# Save Confusion Matrix as Image
# --------------------------------------------------
def save_confusion_matrix(cm, class_names, dataset_name, save_dir):

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(ticks=np.arange(len(class_names)),
               labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)),
               labels=class_names)

    plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                             f"{dataset_name}_confusion_matrix.png"))
    plt.close()


# --------------------------------------------------
# Evaluate Dataset and Save Everything
# --------------------------------------------------
def evaluate_and_save(model, dataset, dataset_name, save_dir):

    print(f"\n==============================")
    print(f"Evaluating: {dataset_name}")
    print(f"==============================")

    dataset.reset()

    # Predictions
    predictions = model.predict(dataset, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = dataset.classes

    class_names = list(dataset.class_indices.keys())

    # --------------------------------------------------
    # Overall Accuracy
    # --------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, dataset_name, save_dir)

    # --------------------------------------------------
    # Per-Class Accuracy
    # (Diagonal / row sum)
    # --------------------------------------------------
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)

    # Avoid division by zero
    class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros_like(class_correct, dtype=float),
        where=class_totals != 0
    )

    per_class_df = pd.DataFrame({
        "class": class_names,
        "total_samples": class_totals,
        "correct_predictions": class_correct,
        "class_accuracy": class_accuracy
    })

    per_class_df.to_csv(
        os.path.join(save_dir,
                     f"{dataset_name}_per_class_accuracy.csv"),
        index=False
    )

    # --------------------------------------------------
    # Classification Report (Precision / Recall / F1)
    # --------------------------------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(
        os.path.join(save_dir,
                     f"{dataset_name}_classification_report.csv")
    )

    # --------------------------------------------------
    # Overall Metrics Table
    # --------------------------------------------------
    overall_metrics = {
        "accuracy": [accuracy],
        "macro_precision": [report_dict["macro avg"]["precision"]],
        "macro_recall": [report_dict["macro avg"]["recall"]],
        "macro_f1": [report_dict["macro avg"]["f1-score"]],
        "weighted_precision": [report_dict["weighted avg"]["precision"]],
        "weighted_recall": [report_dict["weighted avg"]["recall"]],
        "weighted_f1": [report_dict["weighted avg"]["f1-score"]],
    }

    overall_df = pd.DataFrame(overall_metrics)

    overall_df.to_csv(
        os.path.join(save_dir,
                     f"{dataset_name}_overall_metrics.csv"),
        index=False
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    model_path = r"results\model enhanced2.keras"

    train_dir = r"Final-DATA\KNEE-main-images-stablized\train"
    val_dir   = r"Final-DATA\KNEE-main-images-stablized\val"
    test_dir  = r"Final-DATA\KNEE-main-images-stablized\test"

    save_dir = "evaluation_results"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Load Model
    # -------------------------
    model = keras.models.load_model(model_path)

    # -------------------------
    # Data Generators (NO SHUFFLE!)
    # -------------------------
    data_gen = ImageDataGenerator(rescale=1./255)

    train_data = data_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    val_data = data_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    test_data = data_gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    # -------------------------
    # Evaluate All Datasets
    # -------------------------
    evaluate_and_save(model, train_data, "TRAIN", save_dir)
    evaluate_and_save(model, val_data, "VAL", save_dir)
    evaluate_and_save(model, test_data, "TEST", save_dir)

    print("\nAll evaluation files saved in 'evaluation_results' folder.")


if __name__ == "__main__":
    main()
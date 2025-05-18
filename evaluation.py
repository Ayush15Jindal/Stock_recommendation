from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: trained model object
        X_test: test features
        y_test: true labels
    
    Prints:
        Accuracy, classification report, confusion matrix
    """
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from NNClassifier import *
from data_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model_name", type=str, default="crossroad")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nepochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--scaling_flag", type=eval, default=True)
parser.add_argument("--load", default=False, type=eval)
args = parser.parse_args()
print(args)


# Generazione dei dati di esempio
X_train, y_train = load_train_data(args.model_name)
X_val, y_val = load_test_data(args.model_name)

# Creazione dei DataLoader per il training e la validazione
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

if args.model_name == 'navigator':
    nclasses = 4
else:
    nclasses = 3
print('---nclasses = ', nclasses)
# Inizializzazione del modello, ottimizzatore e funzione di perdita
model = TrajectoryClassifier1D(nclasses)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Funzione per il ciclo di allenamento
def train_model(model, train_loader, val_loader, criterion, optimizer):
    for epoch in range(args.nepochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Ciclo di training
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()
            
            # Calcolo della loss cumulativa e accuratezza
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        # Validazione del modello
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        # Stampa dei risultati per epoca
        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        print(f'Epoch [{epoch+1}/{args.nepochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
    
    model_path = f'./save/{args.model_name}/trajectory_classifier_model_{args.nepochs}epochs.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Modello salvato in {model_path}')

# Allenamento del modello
if args.load:
    model_path = f'./save/{args.model_name}/trajectory_classifier_model_{args.nepochs}epochs.pth'
    model.load_state_dict(torch.load(model_path))
else:
    train_model(model, train_loader, val_loader, criterion, optimizer)

def test_model(model, test_loader, criterion):
    model.eval()  # Impostiamo il modello in modalità di valutazione
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    # Disabilitiamo il calcolo dei gradienti durante il test per risparmiare memoria e velocizzare il processo
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calcoliamo la perdita totale
            test_loss += loss.item()
            
            # Calcoliamo le predizioni
            _, predicted = torch.max(outputs, 1)
            
            # Confrontiamo le predizioni con le etichette corrette
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    # Calcoliamo l'accuratezza come percentuale delle predizioni corrette sul totale dei campioni
    test_accuracy = 100 * correct_test / total_test

    # Stampiamo la perdita media e l'accuratezza
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return test_accuracy

# Esempio di utilizzo della funzione di test
# Supponiamo che abbiamo un test_loader con dati di test simile al train_loader
test_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Chiamata della funzione di test
test_model(model, test_loader, criterion)

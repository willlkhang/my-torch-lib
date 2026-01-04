import torch
import pickle

class Trainer:
  def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=20, scheduler=None):

    #training device variables group
    self.device = 'cuda' if torch.cuda.is_available else 'cpu'

    self.model = model.to(self.device)

    #data loader variables group
    self.train_loader = train_loader
    self.val_loader = val_loader

    #training variables group
    self.criterion = criterion
    self.optimizer = optimizer
    self.num_epochs = num_epochs

    self.scheduler = scheduler

    #variabels for tracking and plotting
    self.train_losses = []
    self.val_losses = []
    self.train_accuracies = []
    self.val_accuracies = []

    # TensorBoard writer
    self.writer = SummaryWriter()

  """
  This iterates through each epochs to train the model
  it tracks train/val loss and accuracy
  This methods also write log for tensor board for result visulization and comparation
  """
  def fit(self):
    for epoch in range(self.num_epochs):
      train_loss, train_accuracy = self.train()
      val_loss, val_accuracy = self.evaluate()

      if self.scheduler:
                self.scheduler.step()

      # Store loss and accuracy values
      self.train_losses.append(train_loss)
      self.train_accuracies.append(train_accuracy)
      self.val_losses.append(val_loss)
      self.val_accuracies.append(val_accuracy)

      # Log metrics to TensorBoard
      self.writer.add_scalar('Loss/Train', train_loss, epoch)
      self.writer.add_scalar('Loss/Validation', val_loss, epoch)
      self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
      self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
      print(f'Epoch [{epoch+1}/{self.num_epochs}], '
            f' Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,'
            f' Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    self.writer.close() # Close the writer after training is complete

  """
  This class mode the model on training mode, and uses running loss, correct predicting, and total samples
  to track the loss and accuracy of the model

  The main thing of this method is the gradient calculation
  This basically adjusting the (w,b), so for example, y = aw+b. We are finding the best set of (w,b)
  Then, next time a is enterd, the y will be calculated.

  before doing something, we first cleaning the preivous gradient and calculate the new ones
  """
  def train(self):
    run_loss = 0.0
    correct_pred = 0
    total_samples = 0

    self.model.train()  # Set model to training mode

    for images, labels in self.train_loader:
      images, labels = images.to(self.device), labels.to(self.device)

      self.optimizer.zero_grad()
      outputs = self.model(images)
      loss = self.criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()

      # Track loss and accuracy
      run_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      correct_pred += (predicted == labels).sum().item()
      total_samples += labels.size(0)

    # Calculate training loss and accuracy
    train_loss = run_loss / len(self.train_loader)
    train_accuracy = correct_pred / total_samples * 100

    return train_loss, train_accuracy

  """
  This method is used for evaludating the model,
  the main role of this one is
  after training, we will to evaluate it in order to fine-tune it
  Therefore, in this method, there is no gradient calculation
  """
  @torch.no_grad()
  def evaluate(self):
    val_loss = 0.0
    correct_pred = 0
    total_samples = 0

    self.model.eval()  # Set model to evaluation mode
    for images, labels in self.val_loader:
      images, labels = images.to(self.device), labels.to(self.device)

      # Forward pass
      outputs = self.model(images)
      loss = self.criterion(outputs, labels)
      val_loss += loss.item()

      # Track accuracy
      _, predicted = torch.max(outputs, 1)
      correct_pred += (predicted == labels).sum().item()
      total_samples += labels.size(0)

    # Calculate validation loss and accuracy
    val_loss = val_loss / len(self.val_loader)
    val_accuracy = correct_pred / total_samples * 100

    return val_loss, val_accuracy

  def save_model(self, path):
    print(f'Saving {self.model} to {path}')
    with open(path, 'wb') as f:
      pickle.dump(self.model, f)
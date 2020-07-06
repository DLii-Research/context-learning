import numpy as np

class AtrModel:
    def __init__(self, num_tasks=1, task_switch_threshold=0.0, is_dynamic=False, task_add_threshold=0.0, max_num_tasks=1):
        # Parameters
        self.num_tasks = num_tasks
        self.switch_threshold = task_switch_threshold
        self.num_tasks = num_tasks
        self.is_dynamic = is_dynamic
        self.add_threshold = task_add_threshold
        self.max_num_tasks = max_num_tasks
        
        # ATR model information
        self.atr_values = []
        self.context_losses = [0.0 for i in range(num_tasks)]
        self.num_seq_switches = 0
        
        # The defualt hot context
        self.hot_context_idx = 0
        
        # Initialize the ATR values
        self.initialize_atr_values(self.num_tasks)
        
        
    def add_context(self):
        # Increase the number of ATRs
        self.num_tasks += 1
        
        # Add the new ATR value
        self.atr_values.append(self.initial_atr_value())
        self.context_losses.append(0.0)
    
    
    def switch_contexts(self, context_loss):
        """
        Switch to a new task
        """
        print("Switching...")
        # If we have exhausted the context list, pick the one with the lowest loss
        if self.num_seq_switches >= self.num_tasks:
            # Find the context with the lowest loss
            best_fit_idx = self.find_best_fit_context()
            
            # Check if a new context should be added
            if self.is_dynamic and self.num_tasks < self.max_num_tasks and self.should_add_context(context_loss, best_fit_idx):
                # Add the new context
                self.add_context()
                
                # Switch to the new context
                self.hot_context_idx = self.num_tasks - 1
                print(f"Task switched (added new task): {self.hot_context_idx}")
                
            else:
                # Change to the new context
                self.hot_context_idx = best_fit_idx
                print(f"Task switched (found best): {self.hot_context_idx}")

                # Update the ATR value
                # @WARN This is probably not the best way to do this... but it works, so...
                self.atr_values[self.hot_context_idx] = self.context_losses[self.hot_context_idx]

        else:
            self.next_context()
            print(f"Task switched to: {self.hot_context_idx}")
    
    
    def update_and_switch(self, context_loss, dynamic_switch=True):
        """Perform an update on the ATR model"""
        
        # Do we need to switch contexts?
        if dynamic_switch and self.should_switch(context_loss):
            
            # Update the tracked context loss
            self.context_losses[self.hot_context_idx] = context_loss
            
            # Count the sequential sequence switches
            self.num_seq_switches += 1
            
            # Switch to the new context
            self.switch_contexts(context_loss)
            
            # A switch occurred, no ATR updates happened
            return False
        
        # Update the ATR values
        self.update_atr_value(context_loss)
        
        # Reset the sequential switch counter
        self.num_seq_switches = 0
        
        # No switch occurred, ATR was updated
        return True


    def get_hot_context():
        """Get the active context index"""
        return self.hot_context_idx
    
    
    def set_hot_context(self, hot_context_idx):
        """Set the active context index"""
        if hot_context_idx not in range(self.num_tasks):
            raise IndexError("The given hot context index does not exist")
        self.hot_context_idx = hot_context_idx
        
    
    def next_context(self):
        """Switch to the next sequential context"""
        self.set_hot_context((self.hot_context_idx + 1) % self.num_tasks)
        return self.hot_context_idx
    
    # Overridable ---------------------------------------------------------------------------------
    
    def initial_atr_value(self):
        """Determine the initial value for an ATR"""
        return 1.0
    
    def initialize_atr_values(self, num_tasks):
        """Initialize the `atr_values` parameter"""
        self.atr_values = [self.initial_atr_value() for i in range(num_tasks)]
        
    def update_atr_value(self, context_loss):
        """Update the ATR value"""
        self.atr_values[self.hot_context] = context_loss
    
    def find_best_fit_context(self):
        """Locate the context index with the best fit"""
        # return np.argmax(np.subtract(self.atr_values, self.context_losses))
        # return np.argmin(np.abs(np.subtract(self.context_losses, self.atr_values)))
        return np.argmax(np.subtract(self.atr_values, self.context_losses))
    
    def should_switch(self, context_loss):
        """Determine if the context should switch"""
        delta = self.atr_values[self.hot_context_idx] - context_loss
        return delta < self.switch_threshold
    
    def should_add_context(self, context_loss, best_fit_context_idx):
        """Determine if a new context should be added"""
        print("Best context loss...", self.context_losses[best_fit_context_idx])
        return self.atr_values[self.hot_context_idx] - self.context_losses[best_fit_context_idx] < self.add_threshold
    
    
class AtrMovingAverage(AtrModel):
    def update_atr_value(self, context_loss):
        self.atr_values[self.hot_context_idx] = (self.atr_values[self.hot_context_idx] + context_loss) / 2.0
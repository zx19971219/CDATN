import logging


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, stop_criterion, patience=5):
        self.stop_criterion = stop_criterion
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, curr_score, model):
        if self.best_score is None:
            logging.info('Validation %s (%.6f) initialized. Saving model ...' % (self.stop_criterion, curr_score))
            self.best_score = curr_score
            model.save()
            return False

        elif curr_score > self.best_score:
            logging.info('Validation %s: %.6f --> %.6f.  Saving model ...'
                         % (self.stop_criterion, self.best_score, curr_score))
            self.best_score = curr_score
            model.save()

            self.counter = 0
            return False

        else:
            self.counter += 1
            logging.info('The best validation %s is %.6f. EarlyStopping counter: %i out of %i' %
                         (self.stop_criterion, self.best_score, self.counter, self.patience))
            if self.counter >= self.patience:
                logging.info("Early stop!")
                return True
            else:
                return False



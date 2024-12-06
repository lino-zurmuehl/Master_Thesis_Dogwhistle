import pandas as pd
import os
from datetime import datetime

class LabelReviewPipeline:
    def __init__(self, input_file="vetting_instances/potential_wrong_label.csv", batch_size=10):
        """Initialize the review pipeline."""
        self.input_file = input_file
        self.batch_size = batch_size
        self.reviewed_file = "vetting_instances/reviewed_labels.csv"
        self.load_data()
    
    def load_data(self):
        """Load the data and initialize tracking of reviewed items."""
        # Load potential wrong labels
        self.data = pd.read_csv(self.input_file)
        
        # Load or create reviewed data
        if os.path.exists(self.reviewed_file):
            self.reviewed = pd.read_csv(self.reviewed_file)
        else:
            self.reviewed = pd.DataFrame(columns=['text', 'dog_whistles', 'given_label', 
                                                'suggested_label', 'final_label', 
                                                'review_date', 'reviewer_notes'])
    
    def get_next_batch(self):
        """Get the next batch of unreviewed items."""
        # Filter out already reviewed items
        unreviewed = self.data[~self.data['text'].isin(self.reviewed['text'])]
        return unreviewed.head(self.batch_size)
    
    def review_batch(self):
        """Interactive review process for a batch of items."""
        batch = self.get_next_batch()
        
        if len(batch) == 0:
            print("No more items to review!")
            return
        
        reviewed_items = []
        
        for idx, row in batch.iterrows():
            print("\n" + "="*80)
            print(f"\nText: {row['text']}")
            print(f"Dog Whistle Category: {row['dog_whistles']}")
            print(f"Current Label: {row['given_label']}")
            print(f"Suggested Label: {row['suggested_label']}")
            
            while True:
                decision = input("\nDo you want to change the label? (y/n): ").lower()
                if decision in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")
            
            if decision == 'y':
                while True:
                    try:
                        new_label = int(input("Enter new label (0 or 1): "))
                        if new_label in [0, 1]:
                            break
                        print("Please enter 0 or 1")
                    except ValueError:
                        print("Please enter a valid number (0 or 1)")
            else:
                new_label = row['given_label']
            
            notes = input("Enter any notes (optional): ").strip()
            
            reviewed_items.append({
                'text': row['text'],
                'dog_whistles': row['dog_whistles'],
                'given_label': row['given_label'],
                'suggested_label': row['suggested_label'],
                'final_label': new_label,
                'review_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'reviewer_notes': notes
            })
        
        # Add reviewed items to the reviewed DataFrame
        new_reviews = pd.DataFrame(reviewed_items)
        self.reviewed = pd.concat([self.reviewed, new_reviews], ignore_index=True)
        
        # Save updated reviews
        self.reviewed.to_csv(self.reviewed_file, index=False)
        
        print(f"\nBatch review completed! Reviewed {len(batch)} items.")
        print(f"Total reviewed so far: {len(self.reviewed)}")
    
    def get_review_statistics(self):
        """Get statistics about the review process."""
        if len(self.reviewed) == 0:
            return "No reviews completed yet."
        
        total_reviewed = len(self.reviewed)
        labels_changed = sum(self.reviewed['given_label'] != self.reviewed['final_label'])
        
        stats = f"""
Review Statistics:
-----------------
Total items reviewed: {total_reviewed}
Labels changed: {labels_changed} ({(labels_changed/total_reviewed*100):.1f}%)
Remaining to review: {len(self.data) - total_reviewed}
        """
        return stats

# Example usage
if __name__ == "__main__":
    pipeline = LabelReviewPipeline()
    
    while True:
        print("\n1. Review next batch")
        print("2. Show statistics")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            pipeline.review_batch()
        elif choice == "2":
            print(pipeline.get_review_statistics())
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

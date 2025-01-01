import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import os
import itertools
from urllib.parse import quote

# Configuration
BASE_URL = "https://api.open.fec.gov/v1/schedules/schedule_a/"

@dataclass
class Contributor:
    """Represents a political contributor with name and employer information"""
    name: str
    employer: str

@dataclass
class ContributionCluster:
    """Represents a group of related contributions within a time window"""
    committee_id: str
    committee_name: str
    contributions: List[Dict]
    start_date: datetime
    end_date: datetime
    total_amount: float
    contributors: Set[str]

class FECContributionAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with an API key"""
        self.api_key = api_key
        self.params = {
            'api_key': api_key,
            'sort_hide_null': False,
            'sort_nulls_last': False,
            'sort': '-contribution_receipt_date',
            'is_individual': True
        }

    def get_contributor_data(self, contributor: Contributor, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch all contributions for a given contributor within the date range
        
        Args:
            contributor: Contributor object containing name and employer
            start_date: Start date in MM/DD/YYYY format
            end_date: End date in MM/DD/YYYY format
            
        Returns:
            List of contribution dictionaries
        """
        params = self.params.copy()
        params.update({
            'contributor_name': contributor.name,
            'contributor_employer': contributor.employer,
            'min_date': start_date,
            'max_date': end_date
        })
        
        all_contributions = []
        page = 1
        
        while True:
            params['page'] = page
            response = requests.get(BASE_URL, params=params)
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            
            data = response.json()
            results = data.get('results', [])
            if not results:
                break
                
            all_contributions.extend(results)
            page += 1
            
            # Check if we've reached the last page
            if page > data['pagination']['pages']:
                break
                
        return all_contributions

    def find_contribution_clusters(
        self,
        contributors: List[Contributor],
        start_date: str,
        end_date: str,
        time_window_days: int,
        min_contributors: int = 2
    ) -> List[ContributionCluster]:
        """
        Find clusters of contributions from the specified contributors within a time window
        
        Args:
            contributors: List of Contributor objects to analyze
            start_date: Start date in MM/DD/YYYY format
            end_date: End date in MM/DD/YYYY format
            time_window_days: Maximum days between contributions to be considered a cluster
            min_contributors: Minimum number of contributors required for a cluster
            
        Returns:
            List of ContributionCluster objects
        """
        # Collect all contributions for all contributors
        all_contributions = []
        for contributor in contributors:
            contributions = self.get_contributor_data(contributor, start_date, end_date)
            all_contributions.extend(contributions)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_contributions)
        if df.empty:
            return []

        # Convert date strings to datetime objects
        df['contribution_receipt_date'] = pd.to_datetime(df['contribution_receipt_date'])

        # Group by committee
        clusters = []
        for committee_id, committee_group in df.groupby('committee_id'):
            committee_name = committee_group.iloc[0]['committee']['name']
            
            # Sort contributions by date
            committee_group = committee_group.sort_values('contribution_receipt_date')
            
            # Find clusters within the time window
            current_cluster = []
            for _, row in committee_group.iterrows():
                if not current_cluster or (
                    row['contribution_receipt_date'] - 
                    pd.to_datetime(current_cluster[-1]['contribution_receipt_date'])
                ).days <= time_window_days:
                    current_cluster.append(row.to_dict())
                else:
                    # Process the completed cluster if it meets criteria
                    if len(set(c['contributor_name'] for c in current_cluster)) >= min_contributors:
                        clusters.append(self._create_cluster(current_cluster, committee_id, committee_name))
                    current_cluster = [row.to_dict()]
            
            # Process the last cluster
            if len(current_cluster) >= min_contributors:
                clusters.append(self._create_cluster(current_cluster, committee_id, committee_name))

        return clusters

    def _create_cluster(self, contributions: List[Dict], committee_id: str, committee_name: str) -> ContributionCluster:
        """Create a ContributionCluster object from a list of contributions"""
        dates = [pd.to_datetime(c['contribution_receipt_date']) for c in contributions]
        return ContributionCluster(
            committee_id=committee_id,
            committee_name=committee_name,
            contributions=contributions,
            start_date=min(dates),
            end_date=max(dates),
            total_amount=sum(c['contribution_receipt_amount'] for c in contributions),
            contributors=set(c['contributor_name'] for c in contributions)
        )

def main():
    """Example usage of the FEC Contribution Analyzer"""
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.environ['FEC_API_KEY']
    analyzer = FECContributionAnalyzer(api_key)
    
    # Define contributors to analyze
    contributors = [
         Contributor(
                name="Sundar Pichai",
                employer="Google"),
            Contributor(
                name="Kent Walker",
                employer="Google"),
            Contributor(
                name="Thomas Kurian",
                employer="Google"),
            Contributor(
                name="Jen Fitzpatrick",
                employer="Google"),
            Contributor(
                name="Rick Osterloh",
                employer="Google"),
            Contributor(
                name="Prabhakar Raghavan",
                employer="Google"),
            Contributor(
                name="Lorraine Twohill",
                employer="Google"),
            Contributor(
                name="Corey DuBrowa",
                employer="Google"),
            Contributor(
                name="Neal Mohan",
                employer="Google"),
            Contributor(
                name="Anat Ashkenazi",
                employer="Google"),
            Contributor(
                name="Jeff Dean",
                employer="Google"),
            Contributor(
                name="Ruth Porat",
                employer="Google"),
        Contributor(name="Mark Zuckerberg", employer="Meta"),
        # Add more contributors as needed
    ]
    
    # Find contribution clusters
    clusters = analyzer.find_contribution_clusters(
        contributors=contributors,
        start_date="01/01/2020",
        end_date="12/31/2024",
        time_window_days=30,
        min_contributors=2
    )
    
    # Print results
    for i, cluster in enumerate(clusters, 1):
        print(f"\nCluster {i}:")
        print(f"Committee: {cluster.committee_name}")
        print(f"Date Range: {cluster.start_date.date()} to {cluster.end_date.date()}")
        print(f"Total Amount: ${cluster.total_amount:,.2f}")
        print("Contributors:")
        for contributor in cluster.contributors:
            print(f"- {contributor}")
        print("Individual Contributions:")
        for contrib in cluster.contributions:
            print(f"- {contrib['contributor_name']}: ${contrib['contribution_receipt_amount']:,.2f} "
                  f"on {contrib['contribution_receipt_date']}")

if __name__ == "__main__":
    main()
import requests
import feedparser
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

BASE_URL = "http://export.arxiv.org/api/query"

@dataclass
class Paper:
    """Standardized paper format for arXiv metadata"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    url: str
    pdf_url: str
    source: str = "arxiv"
    doi: Optional[str] = None
    updated_date: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'paper_id': self.paper_id,
            'title': self.title.replace('\n', ' ').strip(),
            'authors': ', '.join(self.authors),
            'abstract': self.abstract.replace('\n', ' ').strip(),
            'published_date': self.published_date.isoformat(),
            'url': self.url,
            'pdf_url': self.pdf_url,
            'categories': self.categories,
            'doi': self.doi
        }

def search_paper(
    query: str = "",
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    year: Optional[int] = None,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "submittedDate",
    sort_order: str = "descending"
) -> List[Paper]:
    """
    Advanced search for arXiv using specific field prefixes and year filtering.
    """
    query_parts = []
    
    # Add keyword search
    if query:
        query_parts.append(f"all:{query}")
    
    # Add field-specific filters
    if title:
        query_parts.append(f"ti:\"{title}\"")
    if author:
        query_parts.append(f"au:\"{author}\"")
    if category:
        query_parts.append(f"cat:{category}")
        
    # Build the base logical query
    final_query = " AND ".join(query_parts) if query_parts else "all:\"\""
    
    # Add Year Filter (using submittedDate range)
    if year:
        date_filter = f"submittedDate:[{year}01010000 TO {year}12312359]"
        final_query = f"({final_query}) AND {date_filter}"

    params = {
        'search_query': final_query,
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        
        papers = []
        for entry in feed.entries:
            # Extract authors
            authors = [a.name for a in entry.authors] if hasattr(entry, 'authors') else []
            
            # Dates
            published = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
            updated = datetime.strptime(entry.updated, '%Y-%m-%dT%H:%M:%SZ')
            
            # PDF Link
            pdf_url = ""
            for link in entry.links:
                if link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
            
            # DOI (if available)
            doi = entry.get('arxiv_doi', None)
            
            papers.append(Paper(
                paper_id=entry.id.split('/')[-1],
                title=entry.title,
                authors=authors,
                abstract=entry.summary,
                published_date=published,
                updated_date=updated,
                url=entry.id,
                pdf_url=pdf_url,
                categories=[t.term for t in entry.tags],
                doi=doi
            ).to_dict())
        return {
            "status_code": 200,
            "data": papers
        }

    except Exception as e:
        return {
            "status_code": 500,
            "data": []
        }

# --- Example usage ---
if __name__ == "__main__":
    # Search for papers about "ALRM" from 2026 specifically in AI category
    results = search_paper(
        query="robotic", 
        year=2026,
        max_results=3,
        author="Example Author"
    )
    
    for p in results["data"]:
        print(f"Title: {p['title']}")
        print(f"Published: {p['published_date']}")
        print(f"URL: {p['url']}\n")
        print(f"Abstract: {p['abstract']}\n")
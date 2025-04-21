# Blue Collar Job Recommender for India

A simple, robust job recommendation system specifically designed for blue collar workers in India. This system takes a user profile as input and provides tailored job recommendations.

## Features

- Uses OpenAI embeddings to create vector representations of jobs
- Uses Google's Gemini Flash 2.0 AI to extract structured user profiles from text descriptions
- Stores job vectors in a FAISS database for efficient similarity search
- Evaluates and refines recommendations based on AI feedback
- Provides explanations for job recommendations
- Supports loading job data from Excel files
- Offers a combined endpoint for one-step recommendation

## Setup

1. Clone this repository
2. Create a `.env` file with the following keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   PORT=9000
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. (Optional) Prepare an Excel file with job data with the following columns:
   - id: unique identifier for each job
   - title: job title
   - description: job description
   - company: company name
   - city: city location
   - state: state location
   - country: country (defaults to India if not specified)
   - salary_range: optional salary information
   - skills_required: comma-separated list of required skills
   - experience_required: years of experience required
   - qualifications_required: comma-separated list of required qualifications
   - contact_info: optional contact information

5. Run the application:
   ```
   python app.py
   ```

## API Endpoints

### Combined Profile Extraction and Job Recommendation

```
POST /recommend
```

Extracts a user profile from text description and returns job recommendations in one step.

**Request Body:**
```json
{
  "text_description": "My name is Raj Kumar, I am 28 years old male from Mumbai, Maharashtra. I speak Hindi and English. I have 3 years of experience as a electrician and I know wiring, circuit testing and electrical repairs. I have completed ITI in Electrical. I'm looking for electrician jobs."
}
```

### Extract Profile Only

```
POST /extract-profile
```

Extracts a structured user profile from a text description.

**Request Body:**
```json
{
  "text_description": "My name is Raj Kumar, I am 28 years old male from Mumbai, Maharashtra. I speak Hindi and English. I have 3 years of experience as a electrician and I know wiring, circuit testing and electrical repairs. I have completed ITI in Electrical. I'm looking for electrician jobs."
}
```

### Get Job Recommendations Only

```
POST /recommend-jobs
```

Gets job recommendations based on a structured user profile.

**Request Body:**
```json
{
  "profile": {
    "name": "Raj Kumar",
    "age": 28,
    "gender": "male",
    "preferred_languages": ["hindi", "english"],
    "preferred_job_roles": ["electrician"],
    "experience": 3,
    "skills": ["wiring", "circuit testing", "electrical repairs"],
    "certifications": [{"name": "ITI Electrical"}],
    "location": {
      "city": "Mumbai",
      "state": "Maharashtra"
    }
  }
}
```

### Upload Jobs Data

```
POST /upload-jobs
```

Upload an Excel file containing job data to rebuild the vector database.

**Request Body:**
- Form data with `file` field containing an Excel file (.xlsx or .xls)

## Example Usage

### One-step recommendation:

```bash
curl -X POST "http://localhost:9000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"text_description": "My name is Raj Kumar, I am 28 years old male from Mumbai, Maharashtra. I speak Hindi and English. I have 3 years of experience as a electrician and I know wiring, circuit testing and electrical repairs. I have completed ITI in Electrical. I am looking for electrician jobs."}'
```

### Upload job data:

```bash
curl -X POST "http://localhost:9000/upload-jobs" \
     -F "file=@/path/to/your/jobs_data.xlsx"
``` 
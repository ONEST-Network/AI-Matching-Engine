# Blue Collar Job Recommender for India

A simple, robust job recommendation system specifically designed for blue collar workers in India. This system takes a user profile as input and provides tailored job recommendations using a combination of AI models (FAISS vector search and Google's Gemini).

## Features

- Uses OpenAI embeddings to create vector representations of jobs
- Uses Google's Gemini Flash 2.0 AI for recommendations and explanations
- Stores job vectors in a FAISS database for efficient similarity search
- Combines FAISS semantic search with LLM-based ranking for accurate recommendations
- Filters jobs by gender, location, experience, and language requirements
- Provides explanations for job recommendations
- Supports loading job data from Excel files

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
4. Run the application:
   ```
   python app.py
   ```

## API Endpoints

### Job Recommendation Endpoint

```
POST /recommend
```

The main endpoint for job recommendations. Takes a JSON user profile and returns personalized job recommendations.

**Request Body:**
```json
{
  "name": "Raj Kumar",
  "gender": "male",
  "age": 28,
  "location": {
    "city": "Mumbai",
    "state": "Maharashtra"
  },
  "qualifications": ["ITI Electrical"],
  "skills": ["wiring", "circuit testing", "electrical repairs"],
  "experience": 3,
  "past_experiences": ["Worked as electrician at BuildTech"],
  "preferred_job_roles": ["electrician"],
  "preferred_languages": ["hindi", "english"]
}
```

**Note:** The `preferred_languages` field is optional. If provided, the system will filter jobs that require languages the user doesn't know.

**Response:**
```json
{
  "user_profile": {
    "name": "Raj Kumar",
    "age": 28,
    "gender": "male",
    "preferred_languages": ["hindi", "english"],
    "preferred_job_roles": ["electrician"],
    "experience": 3,
    "skills": ["wiring", "circuit testing", "electrical repairs"],
    "certifications": [],
    "location": {
      "city": "Mumbai",
      "state": "Maharashtra",
      "country": "India"
    },
    "past_experiences": ["Worked as electrician at BuildTech"],
    "qualifications": ["ITI Electrical"]
  },
  "recommended_jobs": [
    {
      "id": "1",
      "title": "Electrician",
      "description": "Installation and maintenance of electrical systems in residential and commercial buildings.",
      "company": "BuildTech Solutions",
      "location": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "country": "India"
      },
      "salary_range": "₹15,000 - ₹25,000 per month",
      "skills_required": ["wiring", "circuit testing", "troubleshooting", "electrical repairs"],
      "experience_required": 2,
      "qualifications_required": ["ITI Electrical"],
      "contact_info": null
    }
  ],
  "recommendation_reason": "The Electrician job at BuildTech Solutions is an excellent match for your profile as it's in your city of Mumbai, aligns with your experience level, and requires your specific electrical skills and ITI qualification."
}
```

### Health Check Endpoint

```
GET /
```

Simple endpoint to verify that the API is running.

**Response:**
```json
{
  "status": "online",
  "message": "Blue Collar Job Recommender API is running",
  "jobs_loaded": 10
}
```

## Example Usage

### Job recommendation:

```bash
curl -X POST "http://localhost:9000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Raj Kumar",
       "gender": "male",
       "age": 28,
       "location": {
         "city": "Mumbai",
         "state": "Maharashtra"
       },
       "qualifications": ["ITI Electrical"],
       "skills": ["wiring", "circuit testing", "electrical repairs"],
       "experience": 3,
       "past_experiences": ["Worked as electrician at BuildTech"],
       "preferred_job_roles": ["electrician"]
     }'
```

### Health check:

```bash
curl "http://localhost:9000/"
```

## Swagger Documentation

The API includes Swagger documentation accessible at:
- Swagger UI: http://localhost:9000/docs
- ReDoc: http://localhost:9000/redoc
- OpenAPI JSON: http://localhost:9000/openapi.json 

import os
from dotenv import load_dotenv
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np
import pandas as pd
from pathlib import Path

# LangChain and OpenAI for embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Google's Gemini for profile extraction
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", 9000))

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize OpenAI
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings()

# Initialize FastAPI
app = FastAPI(
    title="Blue Collar Job Recommender",
    description="""
    An API for recommending blue collar jobs to workers in India based on their skills, qualifications, and preferences.
    
    ## Features
    
    * Extract structured user profiles from text descriptions
    * Get personalized job recommendations based on user profiles
    * Filter jobs by location, gender, qualifications, and more
    * Upload job data via Excel files
    
    ## Usage
    
    The main endpoint is `/recommend` which accepts a JSON user profile and returns matched jobs.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums and data models
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class Language(str, Enum):
    HINDI = "hindi"
    ENGLISH = "english"
    TAMIL = "tamil"
    TELUGU = "telugu"
    BENGALI = "bengali"
    MARATHI = "marathi"
    GUJARATI = "gujarati"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    PUNJABI = "punjabi"
    ODIA = "odia"
    URDU = "urdu"

class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Driving License",
                "issuer": "RTO Maharashtra",
                "year": 2020
            }
        }

class Location(BaseModel):
    city: str
    state: str
    country: str = "India"
    
    class Config:
        schema_extra = {
            "example": {
                "city": "Mumbai",
                "state": "Maharashtra",
                "country": "India"
            }
        }

class UserProfile(BaseModel):
    name: str
    age: int
    gender: Gender
    preferred_languages: List[Language] = []
    preferred_job_roles: List[str]
    experience: int
    skills: List[str]
    certifications: List[Certification] = []
    location: Location
    past_experiences: List[str] = []
    qualifications: List[str] = []
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Raj Kumar",
                "age": 25,
                "gender": "male",
                "preferred_languages": ["hindi", "english"],
                "preferred_job_roles": ["Delivery Executive", "Driver"],
                "experience": 2,
                "skills": ["Driving", "Navigation", "Customer Service"],
                "certifications": [{"name": "Driving License", "issuer": "RTO Maharashtra", "year": 2020}],
                "location": {"city": "Mumbai", "state": "Maharashtra"},
                "past_experiences": ["Food Delivery at Swiggy"],
                "qualifications": ["12th Pass"]
            }
        }

class Job(BaseModel):
    id: str
    title: str
    description: str
    company: str
    location: Location
    salary_range: Optional[str] = None
    skills_required: List[str] = []
    experience_required: Optional[int] = None
    qualifications_required: List[str] = []
    contact_info: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "42",
                "title": "Delivery Executive",
                "description": "Responsible for timely delivery of packages across the city",
                "company": "Quick Delivery Services",
                "location": {"city": "Mumbai", "state": "Maharashtra"},
                "salary_range": "₹12,000 - ₹18,000 per month",
                "skills_required": ["Driving", "Navigation", "Time Management"],
                "experience_required": 1,
                "qualifications_required": ["10th Pass", "Valid Driving License"],
                "contact_info": "hr@quickdelivery.com"
            }
        }

class UserProfileInput(BaseModel):
    text_description: str
    
    class Config:
        schema_extra = {
            "example": {
                "text_description": "My name is Raj Kumar, I am 25 years old male from Mumbai, Maharashtra. I have completed 12th standard and have 2 years of experience as a delivery person with Swiggy. I can speak Hindi and English. I have a driving license and am looking for delivery or driver jobs."
            }
        }

class JobRecommendationRequest(BaseModel):
    profile: UserProfile
    
    class Config:
        schema_extra = {
            "example": {
                "profile": {
                    "name": "Raj Kumar",
                    "age": 25,
                    "gender": "male",
                    "preferred_languages": ["hindi", "english"],
                    "preferred_job_roles": ["Delivery Executive", "Driver"],
                    "experience": 2,
                    "skills": ["Driving", "Navigation", "Customer Service"],
                    "certifications": [{"name": "Driving License", "issuer": "RTO Maharashtra", "year": 2020}],
                    "location": {"city": "Mumbai", "state": "Maharashtra"},
                    "past_experiences": ["Food Delivery at Swiggy"],
                    "qualifications": ["12th Pass"]
                }
            }
        }

class JobRecommendationResponse(BaseModel):
    user_profile: UserProfile
    recommended_jobs: List[Job]
    recommendation_reason: str
    
    class Config:
        schema_extra = {
            "example": {
                "user_profile": {
                    "name": "Raj Kumar",
                    "age": 25,
                    "gender": "male",
                    "preferred_languages": ["hindi", "english"],
                    "preferred_job_roles": ["Delivery Executive", "Driver"],
                    "experience": 2,
                    "skills": ["Driving", "Navigation", "Customer Service"],
                    "certifications": [{"name": "Driving License", "issuer": "RTO Maharashtra", "year": 2020}],
                    "location": {"city": "Mumbai", "state": "Maharashtra"},
                    "past_experiences": ["Food Delivery at Swiggy"],
                    "qualifications": ["12th Pass"]
                },
                "recommended_jobs": [
                    {
                        "id": "42",
                        "title": "Delivery Executive",
                        "description": "Responsible for timely delivery of packages across the city",
                        "company": "Quick Delivery Services",
                        "location": {"city": "Mumbai", "state": "Maharashtra"},
                        "salary_range": "₹12,000 - ₹18,000 per month",
                        "skills_required": ["Driving", "Navigation", "Time Management"],
                        "experience_required": 1,
                        "qualifications_required": ["10th Pass", "Valid Driving License"],
                        "contact_info": "hr@quickdelivery.com"
                    }
                ],
                "recommendation_reason": "The jobs perfectly match your location in Mumbai and your experience as a delivery person. Your driving skills and qualifications meet the requirements for these positions."
            }
        }

class DirectRecommendationRequest(BaseModel):
    text_description: str
    
    class Config:
        schema_extra = {
            "example": {
                "text_description": "My name is Raj Kumar, I am 25 years old male from Mumbai, Maharashtra. I have completed 12th standard and have 2 years of experience as a delivery person with Swiggy. I can speak Hindi and English. I have a driving license and am looking for delivery or driver jobs."
            }
        }

# Sample blue collar job data - used as fallback if no Excel file is found
SAMPLE_JOBS = [
    {
        "id": "1",
        "title": "Electrician",
        "description": "Installation and maintenance of electrical systems in residential and commercial buildings.",
        "company": "BuildTech Solutions",
        "location": {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
        "salary_range": "₹15,000 - ₹25,000 per month",
        "skills_required": ["wiring", "circuit testing", "troubleshooting", "electrical repairs"],
        "experience_required": 2,
        "qualifications_required": ["ITI Electrical"]
    },
    {
        "id": "2",
        "title": "Plumber",
        "description": "Installation and repair of water supply lines, drainage systems, and fixtures.",
        "company": "Urban Utilities",
        "location": {"city": "Delhi", "state": "Delhi", "country": "India"},
        "salary_range": "₹12,000 - ₹20,000 per month",
        "skills_required": ["pipe fitting", "leak repair", "fixture installation"],
        "experience_required": 1
    },
    {
        "id": "3",
        "title": "Construction Worker",
        "description": "General labor at construction sites including material handling and site preparation.",
        "company": "MegaConstruct India",
        "location": {"city": "Bangalore", "state": "Karnataka", "country": "India"},
        "salary_range": "₹10,000 - ₹18,000 per month",
        "skills_required": ["material handling", "basic construction", "physical strength"]
    },
    {
        "id": "4",
        "title": "Delivery Driver",
        "description": "Delivering packages and goods to customers in a timely and safe manner.",
        "company": "SpeedDelivery",
        "location": {"city": "Chennai", "state": "Tamil Nadu", "country": "India"},
        "salary_range": "₹12,000 - ₹22,000 per month",
        "skills_required": ["driving", "navigation", "time management"],
        "experience_required": 1,
        "qualifications_required": ["Valid driving license"]
    },
    {
        "id": "5",
        "title": "Factory Worker",
        "description": "Operating machinery and assembly line work in manufacturing facility.",
        "company": "Tata Motors",
        "location": {"city": "Pune", "state": "Maharashtra", "country": "India"},
        "salary_range": "₹12,000 - ₹18,000 per month",
        "skills_required": ["machine operation", "assembly line work", "quality control"]
    },
    {
        "id": "6",
        "title": "Security Guard",
        "description": "Monitoring premises and ensuring safety of people and property.",
        "company": "SecureForce India",
        "location": {"city": "Hyderabad", "state": "Telangana", "country": "India"},
        "salary_range": "₹10,000 - ₹15,000 per month",
        "skills_required": ["surveillance", "security protocols", "emergency response"]
    },
    {
        "id": "7",
        "title": "Carpenter",
        "description": "Construction and repair of wooden structures and fixtures.",
        "company": "WoodWorks Construction",
        "location": {"city": "Jaipur", "state": "Rajasthan", "country": "India"},
        "salary_range": "₹15,000 - ₹25,000 per month",
        "skills_required": ["woodworking", "measuring", "cutting", "furniture making"],
        "experience_required": 2
    },
    {
        "id": "8",
        "title": "Cook/Chef Assistant",
        "description": "Food preparation and kitchen assistance in restaurants or catering services.",
        "company": "Taj Hotels",
        "location": {"city": "Kolkata", "state": "West Bengal", "country": "India"},
        "salary_range": "₹12,000 - ₹20,000 per month",
        "skills_required": ["cooking", "food preparation", "kitchen hygiene"]
    },
    {
        "id": "9",
        "title": "Textile Worker",
        "description": "Operating textile machinery and quality control of fabric production.",
        "company": "Arvind Mills",
        "location": {"city": "Ahmedabad", "state": "Gujarat", "country": "India"},
        "salary_range": "₹10,000 - ₹16,000 per month",
        "skills_required": ["machine operation", "quality control", "basic textile knowledge"]
    },
    {
        "id": "10",
        "title": "Housekeeping Staff",
        "description": "Cleaning and maintaining hotels, offices, or residential buildings.",
        "company": "CleanPro Services",
        "location": {"city": "Goa", "state": "Goa", "country": "India"},
        "salary_range": "₹8,000 - ₹15,000 per month",
        "skills_required": ["cleaning", "organization", "time management"]
    }
]

# Function to load jobs from Excel file
def load_jobs_from_excel(file_path="SyntheticJobsONEST.xlsx"):
    try:
        df = pd.read_excel(file_path)
        jobs_list = []
        
        for index, row in df.iterrows():
            try:
                # Extract skills from Mandatory Requirements field
                skills = [] if pd.isna(row.get('Mandatory Requirements', '')) else str(row['Mandatory Requirements']).split(',')
                skills = [s.strip() for s in skills]
                
                # Extract qualifications from Minimum Qualification and Specialization Required fields
                qualifications = []
                if not pd.isna(row.get('Minimum Qualification', '')):
                    qualifications.append(str(row['Minimum Qualification']).strip())
                if not pd.isna(row.get('Specialization Required', '')):
                    qualifications.append(str(row['Specialization Required']).strip())
                
                # Create salary range string - with proper type checking
                salary_range = None
                
                # Check if Starting Salary contains non-numeric values
                start_salary_numeric = True
                end_salary_numeric = True
                
                if not pd.isna(row.get('Starting Salary', '')):
                    if isinstance(row['Starting Salary'], (int, float)):
                        start_salary = int(row['Starting Salary'])
                    else:
                        # If it's a string like "Freelancers/Commission Based"
                        start_salary_numeric = False
                        salary_range = str(row['Starting Salary'])
                
                if not pd.isna(row.get('Ending Salary', '')):
                    if isinstance(row['Ending Salary'], (int, float)):
                        end_salary = int(row['Ending Salary'])
                    else:
                        end_salary_numeric = False
                
                # Create formatted salary range only if both are numeric
                if start_salary_numeric and end_salary_numeric and not pd.isna(row.get('Starting Salary', '')) and not pd.isna(row.get('Ending Salary', '')):
                    salary_range = f"₹{int(row['Starting Salary'])} - ₹{int(row['Ending Salary'])} per month"
                elif start_salary_numeric and not pd.isna(row.get('Starting Salary', '')) and salary_range is None:
                    salary_range = f"₹{int(row['Starting Salary'])} per month"
                elif not start_salary_numeric and salary_range is None:
                    # Use the string value as is
                    salary_range = str(row['Starting Salary'])
                
                # Extract location
                city = row.get('Location', '') if not pd.isna(row.get('Location', '')) else "Unknown"
                state = row.get('State', '') if not pd.isna(row.get('State', '')) else "Unknown"
                
                # Create location object
                location = {
                    "city": city,
                    "state": state,
                    "country": "India"
                }
                
                # Experience required (use Min Exp)
                experience_required = None
                if not pd.isna(row.get('Min Exp', '')):
                    if isinstance(row['Min Exp'], (int, float)):
                        experience_required = int(row['Min Exp'])
                
                # Create job object
                job = {
                    "id": str(row['SN']),
                    "title": row['Designation'],
                    "description": row['Job Description'] if not pd.isna(row.get('Job Description', '')) else "",
                    "company": row['Company Name'],
                    "location": location,
                    "salary_range": salary_range,
                    "skills_required": skills,
                    "experience_required": experience_required,
                    "qualifications_required": qualifications,
                    "contact_info": None
                }
                
                jobs_list.append(job)
            except Exception as row_error:
                print(f"Error processing row {index}: {str(row_error)}")
                continue
        
        print(f"Successfully loaded {len(jobs_list)} jobs from {file_path}")
        return jobs_list
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        # Fall back to sample jobs if Excel file can't be loaded
        return SAMPLE_JOBS

# Initialize jobs as objects
jobs = []

# Helper function to create FAISS index
def initialize_faiss_index(job_list: List[Job]):
    # Convert jobs to text documents for embedding
    job_texts = []
    metadatas = []
    
    for job in job_list:
        # Create a text representation of the job
        text = (
            f"Job: {job.title}. Company: {job.company}. Location: {job.location.city}, {job.location.state}. "
            f"Description: {job.description}. "
        )
        if job.skills_required:
            skills_str = ", ".join(job.skills_required)
            text += f"Skills: {skills_str}. "
        if job.experience_required:
            text += f"Experience: {job.experience_required} years. "
        if job.qualifications_required:
            quals_str = ", ".join(job.qualifications_required)
            text += f"Qualifications: {quals_str}. "
        if job.salary_range:
            text += f"Salary: {job.salary_range}. "
        
        job_texts.append(text)
        metadatas.append({"id": job.id})
    
    # Create FAISS index
    return FAISS.from_texts(job_texts, embeddings, metadatas=metadatas)

# Initialize FAISS index on startup
faiss_index = None

@app.on_event("startup")
async def startup_event():
    global jobs, faiss_index
    
    # Load jobs from Excel if available, otherwise use sample jobs
    excel_path = "SyntheticJobsONEST.xlsx"
    if Path(excel_path).exists():
        jobs_data = load_jobs_from_excel(excel_path)
    else:
        jobs_data = SAMPLE_JOBS
    
    # Create Job objects
    jobs = [Job(**job) for job in jobs_data]
    
    # Initialize FAISS index
    faiss_index = initialize_faiss_index(jobs)
    print(f"Loaded {len(jobs)} jobs into the recommendation system")

# Extract user profile using Gemini
def extract_user_profile(text_description: str) -> UserProfile:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        Extract a structured user profile from the following text for a blue collar worker in India.
        Return a valid JSON object with these fields:
        
        - name: string (person's name)
        - age: int (person's age)
        - gender: string (one of: male, female, other)
        - preferred_languages: array of strings (languages known by the person)
        - preferred_job_roles: array of strings (job types the person is looking for)
        - experience: int (years of work experience)
        - skills: array of strings (skills the person has)
        - certifications: array of objects with fields: name, issuer (optional), year (optional)
        - location: object with fields: city, state, country
        - past_experiences: array of strings (previous job experiences)
        - qualifications: array of strings (educational qualifications)
        
        Text: {text_description}
        
        Format your response ONLY as a valid JSON object, with no additional text, markdown formatting, or explanation.
        Do not include any backticks or json code block formatting. Just the raw JSON.
        """
        
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "response_mime_type": "application/json",
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Print raw response for debugging
        print("Raw model response:", response.text)
        
        # Clean the response to ensure it's valid JSON
        cleaned_text = response.text.strip()
        
        # Remove any code block markers or extra text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        # Remove any leading/trailing whitespace again after cleaning
        cleaned_text = cleaned_text.strip()
        
        try:
            profile_json = json.loads(cleaned_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
            print(f"Attempted to parse: {cleaned_text}")
            
            # Try to extract JSON from a larger text if it's not pure JSON
            import re
            json_match = re.search(r'({.*})', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    profile_json = json.loads(json_match.group(1))
                except:
                    raise ValueError(f"Failed to extract valid JSON from model response: {cleaned_text}")
            else:
                raise ValueError(f"Failed to extract valid JSON from model response: {cleaned_text}")
        
        # Validate that all required fields are present
        required_fields = ["name", "age", "gender", "preferred_languages", "preferred_job_roles", 
                          "experience", "skills", "location"]
        for field in required_fields:
            if field not in profile_json:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure location has all required fields
        if not all(k in profile_json["location"] for k in ["city", "state"]):
            raise ValueError("Location must have city and state")
        
        # Convert lists to appropriate types if needed - converting language to lowercase first
        profile_json["preferred_languages"] = [Language(lang.lower()) for lang in profile_json["preferred_languages"]]
        
        # Create certifications as objects
        certifications = []
        for cert in profile_json.get("certifications", []):
            certifications.append(Certification(**cert))
        profile_json["certifications"] = certifications
        
        # Set country to India if not specified
        if "country" not in profile_json["location"]:
            profile_json["location"]["country"] = "India"
            
        # Create and return a UserProfile object
        return UserProfile(**profile_json)
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Profile extraction error: {str(e)}")
        print(f"Traceback: {traceback_str}")
        raise HTTPException(status_code=400, detail=f"Failed to extract profile: {str(e)}")

# Evaluate response using the given prompt
def evaluate_response(query: str, response: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        You are an evaluator of AI generated answers. Your task is to investigate whether the given response is logical and helpful to answer the query.\n
        Please judge the response step by step.\n
        First, given the query, context and response, give your binary judgement [Yes or No] if the response has good logic to answer the query in the context of the context.\n
        If the answer is 'Yes', return 'Yes' only.\n
        Second, if the answer of the first step is "No", re-write the query in the way it helps to retrieve more reverent context.\n
        In the case you are re-writing the query, return the re-written query only.\n
        Here is the query: {query}.\n
        Here is the response: {response}.\n
        """
        
        eval_response = model.generate_content(prompt)
        return eval_response.text.strip()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Search for jobs based on user profile using combined FAISS and LLM approach
def search_jobs(user_profile: UserProfile) -> List[Job]:
    global jobs, faiss_index
    
    print(f"\n=== SEARCHING JOBS FOR: {user_profile.name} ===")
    print(f"Gender: {user_profile.gender.lower()}")
    print(f"Location: {user_profile.location.city}, {user_profile.location.state}")
    print(f"Preferred roles: {user_profile.preferred_job_roles}")
    
    # Check if user specified languages
    has_language_preferences = len(user_profile.preferred_languages) > 0
    if has_language_preferences:
        print(f"Languages: {[lang.value for lang in user_profile.preferred_languages]}")
        # Convert user languages to lowercase for easier matching
        user_languages = [lang.value.lower() for lang in user_profile.preferred_languages]
        print(f"User languages: {user_languages}")
    else:
        print("No language preferences specified - skipping language filtering")
    
    # STEP 1: Apply basic filters (gender, location, experience, languages) to pre-filter candidates
    # These are hard constraints that must be satisfied
    basic_filtered_jobs = []
    
    for job in jobs:
        # Check gender compatibility (only filter female-specific jobs)
        is_female_specific = False
        if "female" in job.title.lower():
            is_female_specific = True
        if job.description and ("female only" in job.description.lower() or "women only" in job.description.lower()):
            is_female_specific = True
            
        if is_female_specific and user_profile.gender.lower() != "female":
            print(f"Skipping job {job.id}: Female-specific job, user is {user_profile.gender}")
            continue
            
        # Location filtering (must be in same state)
        if job.location.state.lower() != user_profile.location.state.lower():
            print(f"Skipping job {job.id}: Wrong state: {job.location.state} vs {user_profile.location.state}")
            continue
            
        # Experience filtering (check if user meets minimum experience requirement)
        if job.experience_required is not None and user_profile.experience < job.experience_required:
            print(f"Skipping job {job.id}: Not enough experience. Required: {job.experience_required}, User has: {user_profile.experience}")
            continue
        
        # Language filtering - only if user has specified languages
        if has_language_preferences:
            required_languages = []
            
            # Check job title for language requirements
            if job.title and any(lang in job.title.lower() for lang in ["language", "translator", "interpreter"]):
                # Look for specific language mentions
                for lang in ["hindi", "english", "tamil", "telugu", "marathi", "bengali", "gujarati", "kannada", "malayalam", "punjabi", "odia", "urdu"]:
                    if lang in job.title.lower():
                        required_languages.append(lang)
            
            # Check description for language requirements
            if job.description:
                desc_lower = job.description.lower()
                for lang in ["hindi", "english", "tamil", "telugu", "marathi", "bengali", "gujarati", "kannada", "malayalam", "punjabi", "odia", "urdu"]:
                    if lang in desc_lower:
                        required_languages.append(lang)
            
            # Check qualifications for language requirements
            for qual in job.qualifications_required:
                qual_lower = qual.lower()
                # Check for any language mentions in qualifications
                for lang in ["hindi", "english", "tamil", "telugu", "marathi", "bengali", "gujarati", "kannada", "malayalam", "punjabi", "odia", "urdu"]:
                    if lang in qual_lower:
                        # Check if it's a requirement or just a mention
                        if any(term in qual_lower for term in ["skill", "speak", "communication", "language", "fluent", "proficient", "require", "must", "should"]):
                            required_languages.append(lang)
                        # Special case for qualification strings with "&" - likely a requirement
                        elif "&" in qual_lower and lang in qual_lower:
                            required_languages.append(lang)
            
            # If job requires languages the user doesn't have, skip it
            if required_languages:
                required_languages = list(set(required_languages))  # Remove duplicates
                print(f"Job {job.id} requires languages: {required_languages}")
                
                missing_languages = [lang for lang in required_languages if lang not in user_languages]
                if missing_languages:
                    print(f"Skipping job {job.id}: Requires languages user doesn't know: {missing_languages}")
                    continue
        
        # If all basic filters pass, add to filtered list
        basic_filtered_jobs.append(job)
    
    if has_language_preferences:
        print(f"Found {len(basic_filtered_jobs)} jobs matching basic criteria (gender, location, experience, languages)")
    else:
        print(f"Found {len(basic_filtered_jobs)} jobs matching basic criteria (gender, location, experience)")
    
    if not basic_filtered_jobs:
        print("No jobs matched the basic criteria!")
        return []
    
    # STEP 2: Use FAISS for semantic search based on user profile
    # Create a rich text representation of the user
    user_text = f"""
    Job seeker: {user_profile.name}
    Gender: {user_profile.gender}
    Location: {user_profile.location.city}, {user_profile.location.state}
    Preferred job roles: {', '.join(user_profile.preferred_job_roles)}
    Skills: {', '.join(user_profile.skills)}
    Experience: {user_profile.experience} years
    Qualifications: {', '.join(user_profile.qualifications)}
    Past experiences: {', '.join(user_profile.past_experiences)}
    """
    
    # Get FAISS vector search results using the user text
    if faiss_index:
        try:
            # Use the user text to query the FAISS index
            docs_and_scores = faiss_index.similarity_search_with_score(user_text, k=15)
            
            print(f"FAISS returned {len(docs_and_scores)} semantic matches")
            
            # Extract job IDs and scores from FAISS results
            faiss_job_scores = {}
            for doc, score in docs_and_scores:
                job_id = doc.metadata.get("id")
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + score)  # Transform distance to a 0-1 score
                faiss_job_scores[job_id] = similarity
                print(f"FAISS match - Job ID: {job_id}, Similarity: {similarity:.4f}")
            
            # Filter the basic_filtered_jobs based on FAISS results
            # and add the FAISS score to each job
            semantic_jobs = []
            for job in basic_filtered_jobs:
                if job.id in faiss_job_scores:
                    # This job was semantically matched by FAISS
                    semantic_score = faiss_job_scores[job.id]
                    semantic_jobs.append((job, semantic_score))
            
            # If we have semantic matches, use them
            if semantic_jobs:
                print(f"Found {len(semantic_jobs)} jobs with semantic matching")
                # Sort by semantic score (highest first)
                semantic_jobs.sort(key=lambda x: x[1], reverse=True)
                
                # Take top semantic matches (max 10)
                semantic_filtered_jobs = [job for job, score in semantic_jobs[:10]]
            else:
                # Fall back to basic filtered jobs if no semantic matches
                print("No semantic matches found, falling back to basic filtered jobs")
                semantic_filtered_jobs = basic_filtered_jobs[:10]
        except Exception as e:
            print(f"Error in FAISS search: {str(e)}")
            # Fall back to basic filtered jobs
            semantic_filtered_jobs = basic_filtered_jobs[:10]
    else:
        print("FAISS index not initialized, using only basic filtering")
        semantic_filtered_jobs = basic_filtered_jobs[:10]
    
    print(f"Selected {len(semantic_filtered_jobs)} candidates for LLM ranking")
    
    # STEP 3: Use LLM to rank and explain the matches
    # If we don't have any jobs after filtering, return empty list
    if not semantic_filtered_jobs:
        return []
    
    try:
        # Use LLM to rank the jobs based on user profile
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create a list of job descriptions for the LLM to rank
        job_descriptions = []
        for i, job in enumerate(semantic_filtered_jobs):
            job_text = f"Job {i+1}: {job.title} at {job.company}, {job.location.city}, {job.location.state}. "
            if job.description:
                job_text += f"Description: {job.description[:200]}... "
            if job.skills_required:
                job_text += f"Skills required: {', '.join(job.skills_required)}. "
            if job.qualifications_required:
                job_text += f"Qualifications: {', '.join(job.qualifications_required)}. "
            if job.experience_required is not None:
                job_text += f"Experience: {job.experience_required} years. "
            if job.salary_range:
                job_text += f"Salary: {job.salary_range}. "
            
            job_descriptions.append(job_text)
        
        # Create a detailed prompt for the LLM
        prompt = f"""
        You are a job recommendation system. Rank the following jobs for this user based on how well they match the user's profile.
        Return a ranked list of job indices (1-based) and a brief explanation for each ranking.
        
        User Profile:
        Name: {user_profile.name}
        Gender: {user_profile.gender}
        Location: {user_profile.location.city}, {user_profile.location.state}
        Preferred Roles: {', '.join(user_profile.preferred_job_roles)}
        Skills: {', '.join(user_profile.skills)}
        Experience: {user_profile.experience} years
        Qualifications: {', '.join(user_profile.qualifications)}
        Past Experiences: {', '.join(user_profile.past_experiences)}
        
        Available Jobs:
        {chr(10).join(job_descriptions)}
        
        IMPORTANT: Rank these jobs from most relevant to least relevant for this specific user. 
        For each job, consider:
        1. How well the job matches the user's preferred roles
        2. Location match (exact city is better than just state)
        3. Skill match between user's skills and job requirements
        4. Qualification match
        5. Experience match
        
        Format your response as a valid JSON list of objects with 'job_index' (1-based) and 'reason' fields, like:
        [
            {"job_index": 5, "reason": "Perfect match for delivery role with exact skill match and location"},
            {"job_index": 2, "reason": "Good match for sales role but requires relocation within state"},
            ...
        ]
        
        Return ONLY this JSON with no additional text, markdown formatting, or explanation.
        """
        
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "response_mime_type": "application/json",
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        # Get the LLM's response
        try:
            # Parse the LLM's ranking
            llm_rankings = json.loads(response.text)
            
            print(f"LLM ranked {len(llm_rankings)} jobs")
            
            # Create the final ranked list based on LLM output
            ranked_jobs = []
            for ranking in llm_rankings:
                try:
                    # Convert 1-based index to 0-based
                    job_index = ranking["job_index"] - 1
                    if 0 <= job_index < len(semantic_filtered_jobs):
                        job = semantic_filtered_jobs[job_index]
                        reason = ranking.get("reason", "")
                        print(f"LLM Rank: Job {job.id} - {job.title} - Reason: {reason}")
                        ranked_jobs.append(job)
                except (KeyError, TypeError, IndexError) as e:
                    print(f"Error processing LLM ranking: {str(e)}")
                    continue
            
            # If LLM ranking worked, use it
            if ranked_jobs:
                # Take top 5 from LLM ranking
                return ranked_jobs[:5]
            else:
                # Fall back to semantic filtered jobs
                return semantic_filtered_jobs[:5]
                
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {str(e)}")
            # Fall back to semantic filtered jobs
            return semantic_filtered_jobs[:5]
            
    except Exception as e:
        print(f"Error in LLM ranking: {str(e)}")
        # Fall back to semantic filtered jobs
        return semantic_filtered_jobs[:5]

# Generate recommendation reason
def generate_recommendation_reason(user_profile: UserProfile, recommended_jobs: List[Job]) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create descriptions of the recommended jobs with emphasis on location and gender matches
        job_descriptions = []
        for job in recommended_jobs:
            # Check if job location matches user location
            location_match = "matching your location" if job.location.city.lower() == user_profile.location.city.lower() else (
                "in your state" if job.location.state.lower() == user_profile.location.state.lower() else "in a different region"
            )
            
            job_desc = (
                f"Job: {job.title} at {job.company} in {job.location.city}, {job.location.state} ({location_match}). "
                f"Skills required: {', '.join(job.skills_required)}. "
                f"Experience required: {job.experience_required or 'Not specified'} years. "
                f"Salary: {job.salary_range or 'Not specified'}."
            )
            job_descriptions.append(job_desc)
        
        # Create user profile summary with emphasis on gender and location
        user_summary = (
            f"User: {user_profile.name}, {user_profile.age} years old, {user_profile.gender}. "
            f"Location: {user_profile.location.city}, {user_profile.location.state}. "
            f"Experience: {user_profile.experience} years. "
            f"Skills: {', '.join(user_profile.skills)}. "
            f"Preferred job roles: {', '.join(user_profile.preferred_job_roles)}."
        )
        
        prompt = f"""
        As a job recommendation system for blue collar workers in India, explain in 2-4 sentences why the following jobs
        are being recommended to this user. Focus primarily on location matches and gender suitability, 
        then address skills, experience, and other qualifications.
        
        User Profile:
        {user_summary}
        
        Recommended Jobs:
        {' '.join(job_descriptions)}
        
        Include specific references to how well the jobs match the user's location and other key requirements.
        Keep your response concise (2-4 sentences maximum).
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"Jobs recommended based on your location ({user_profile.location.city}, {user_profile.location.state}), gender ({user_profile.gender}), skills, and experience preferences."

# API endpoints
# @app.post("/extract-profile", response_model=UserProfile, 
#          summary="Extract a structured user profile from text",
#          description="Takes a free-text description of a user and extracts structured information about their skills, qualifications, and preferences.")
# async def extract_profile_endpoint(input_data: UserProfileInput):
#     """
#     Extract a structured user profile from a text description.
    
#     - **text_description**: Free text description of the user's background and preferences
    
#     Returns a structured UserProfile object with all the extracted information.
#     """
#     return extract_user_profile(input_data.text_description)

# @app.post("/recommend-jobs", response_model=JobRecommendationResponse,
#          summary="Get job recommendations from a user profile object",
#          description="Takes a structured user profile and returns job recommendations filtered by location, qualifications, and job preferences.")
# async def recommend_jobs_endpoint(request: JobRecommendationRequest):
#     """
#     Get job recommendations based on a user profile.
    
#     - **profile**: Complete user profile object with demographic info, skills, and preferences
    
#     Returns a list of recommended jobs with an explanation of why they were recommended.
#     """
#     # Get recommended jobs
#     recommended_jobs = search_jobs(request.profile)
    
#     if not recommended_jobs:
#         return JobRecommendationResponse(
#             user_profile=request.profile,
#             recommended_jobs=[],
#             recommendation_reason="No jobs found matching your location and gender requirements. Try broadening your search parameters or check back later for new job postings."
#         )
    
#     # Generate explanation for recommendations
#     recommendation_reason = generate_recommendation_reason(request.profile, recommended_jobs)
    
#     # Evaluate the recommendation
#     evaluation = evaluate_response(
#         f"Finding jobs for {request.profile.name} with skills: {', '.join(request.profile.skills)}",
#         f"Recommended jobs: {', '.join([job.title for job in recommended_jobs])}"
#     )
    
#     # If evaluation suggests a rewrite, use it to search again
#     if evaluation != "Yes":
#         # Use the evaluation as a new query to search again
#         try:
#             model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
#             revised_query = f"""
#             Based on this profile:
#             User: {request.profile.name}, Age: {request.profile.age}, Gender: {request.profile.gender},
#             Skills: {', '.join(request.profile.skills)},
#             Experience: {request.profile.experience} years,
#             Location: {request.profile.location.city}, {request.profile.location.state},
            
#             Generate a better search query to find suitable blue-collar jobs in India.
#             """
            
#             response = model.generate_content(revised_query)
#             new_query = response.text.strip()
            
#             # Search with the new query - but still enforce location/gender filtering
#             new_recommended_jobs = search_jobs(request.profile)
            
#             if new_recommended_jobs:
#                 recommended_jobs = new_recommended_jobs
#                 recommendation_reason = generate_recommendation_reason(request.profile, recommended_jobs)
#         except Exception:
#             # If revision fails, stick with original recommendations
#             pass
    
#     return JobRecommendationResponse(
#         user_profile=request.profile,
#         recommended_jobs=recommended_jobs,
#         recommendation_reason=recommendation_reason
#     )

@app.post("/recommend", response_model=JobRecommendationResponse,
         summary="Get job recommendations from a JSON profile",
         description="The primary endpoint for job recommendations. Takes a JSON user profile and returns personalized job recommendations.")
async def combined_recommendation_endpoint(request_data: dict):
    """
    Get job recommendations based on user profile JSON.
    
    Takes a JSON object with the following fields:
    - **name**: User's full name
    - **gender**: "male", "female", or "other" (case-insensitive)
    - **age**: User's age in years
    - **location**: Object with city and state
    - **preferred_job_roles**: List of job roles the user is interested in
    - **experience**: Years of work experience
    - **skills**: List of user's skills
    - **qualifications**: List of educational qualifications
    - **past_experiences**: List of previous job experiences (optional)
    - **preferred_languages**: List of languages the user knows (optional)
    
    Returns a list of recommended jobs with an explanation of why they were recommended.
    """
    # Make a copy of the request data to modify it
    request_data_copy = request_data.copy()
    
    # Convert gender to lowercase for case-insensitive matching
    if "gender" in request_data_copy and isinstance(request_data_copy["gender"], str):
        request_data_copy["gender"] = request_data_copy["gender"].lower()
    
    # Create UserProfile from the modified data
    try:
        user_profile = UserProfile(**request_data_copy)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid profile data: {str(e)}")
    
    # Get recommended jobs
    recommended_jobs = search_jobs(user_profile)
    
    if not recommended_jobs:
        return JobRecommendationResponse(
            user_profile=user_profile,
            recommended_jobs=[],
            recommendation_reason="No jobs found matching your location and gender requirements. Try broadening your search parameters or check back later for new job postings."
        )
    
    # Generate explanation for recommendations
    recommendation_reason = generate_recommendation_reason(user_profile, recommended_jobs)
    
    # Evaluate the recommendation
    evaluation = evaluate_response(
        f"Finding jobs for {user_profile.name} with skills: {', '.join(user_profile.skills)}",
        f"Recommended jobs: {', '.join([job.title for job in recommended_jobs])}"
    )
    
    # If evaluation suggests a rewrite, use it to search again
    if evaluation != "Yes":
        # Use the evaluation as a new query to search again
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            revised_query = f"""
            Based on this profile:
            User: {user_profile.name}, Age: {user_profile.age}, Gender: {user_profile.gender},
            Skills: {', '.join(user_profile.skills)},
            Experience: {user_profile.experience} years,
            Location: {user_profile.location.city}, {user_profile.location.state},
            
            Generate a better search query to find suitable blue-collar jobs in India.
            """
            
            response = model.generate_content(revised_query)
            new_query = response.text.strip()
            
            # Search with the new query, but still enforce hard requirements
            new_recommended_jobs = search_jobs(user_profile)
            
            if new_recommended_jobs:
                recommended_jobs = new_recommended_jobs
                recommendation_reason = generate_recommendation_reason(user_profile, recommended_jobs)
        except Exception:
            # If revision fails, stick with original recommendations
            pass
    
    return JobRecommendationResponse(
        user_profile=user_profile,
        recommended_jobs=recommended_jobs,
        recommendation_reason=recommendation_reason
    )

# @app.post("/upload-jobs", 
#          summary="Upload job data from Excel file",
#          description="Upload an Excel file containing job listings to replace the existing job database.")
# async def upload_jobs_file(file: UploadFile = File(...)):
#     """
#     Upload Excel file with job data and rebuild the vector database.
    
#     - **file**: Excel file (.xlsx or .xls) containing job listings data
    
#     The file should have columns for job titles, descriptions, locations, qualifications, etc.
#     """
#     global jobs, faiss_index
    
#     if not file.filename.endswith(('.xlsx', '.xls')):
#         raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
#     try:
#         # Save the uploaded file
#         file_path = "jobs_data.xlsx"
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
        
#         # Load jobs from the file
#         jobs_data = load_jobs_from_excel(file_path)
#         jobs = [Job(**job) for job in jobs_data]
        
#         # Rebuild FAISS index
#         faiss_index = initialize_faiss_index(jobs)
        
#         return {"message": f"Successfully uploaded and processed {len(jobs)} jobs"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/", 
        summary="API Health Check",
        description="Simple endpoint to verify that the API is running.")
async def root():
    """
    Health check endpoint.
    
    Returns the current status of the API and the number of jobs in the database.
    """
    return {
        "status": "online", 
        "message": "Blue Collar Job Recommender API is running",
        "jobs_loaded": len(jobs)
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True) 
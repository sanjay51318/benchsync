#!/usr/bin/env python3
"""
Complete FastAPI Backend for Consultant Bench Management System
Integrates with PostgreSQL database for real-time data synchronization
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_, func
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Database imports
from database.professional_connection import SessionLocal, engine
from database.models.professional_models import *

app = FastAPI(title="Consultant Bench Management API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Models (Pydantic)
from pydantic import BaseModel

class OpportunityCreate(BaseModel):
    title: str
    description: str
    client_name: str
    required_skills: List[str]
    experience_level: str
    project_duration: str
    budget_range: str
    start_date: str
    end_date: str

class OpportunityUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    user: Optional[Dict[str, Any]] = None
    session_token: Optional[str] = None
    message: str

class AttendanceRequest(BaseModel):
    user_id: str
    date: str  # YYYY-MM-DD
    status: str  # 'present' | 'absent' | 'half_day' | 'leave'
    check_in_time: Optional[str] = None  # HH:MM
    check_out_time: Optional[str] = None  # HH:MM
    hours_worked: Optional[float] = 8.0
    location: Optional[str] = "office"
    notes: Optional[str] = None

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Simple authentication - checks if consultant exists in database
    In production, this would verify password hashes and implement proper JWT tokens
    """
    try:
        # Check if user exists in consultants table
        consultant = db.query(ConsultantProfile).join(User).filter(
            User.email == request.email
        ).first()
        
        if consultant:
            # Check if password matches (simple check for demo)
            if request.password == "password123":
                # Determine role based on email or other criteria
                role = "admin" if "admin" in request.email.lower() else "consultant"
                
                user_data = {
                    "id": consultant.id,
                    "name": consultant.user.name,
                    "email": consultant.user.email,
                    "role": role,
                    "department": consultant.user.department
                }
                
                # Generate a simple session token (in production, use proper JWT)
                session_token = f"session_{consultant.id}_{datetime.now().timestamp()}"
                
                return LoginResponse(
                    success=True,
                    user=user_data,
                    session_token=session_token,
                    message="Login successful"
                )
            else:
                return LoginResponse(
                    success=False,
                    message="Invalid password"
                )
        else:
            # For demo purposes, create admin user if email contains "admin" and password is correct
            if "admin" in request.email.lower() and request.password == "password123":
                user_data = {
                    "id": "admin_1",
                    "name": "System Administrator",
                    "email": request.email,
                    "role": "admin",
                    "department": "Administration"
                }
                
                session_token = f"session_admin_{datetime.now().timestamp()}"
                
                return LoginResponse(
                    success=True,
                    user=user_data,
                    session_token=session_token,
                    message="Admin login successful"
                )
            else:
                return LoginResponse(
                    success=False,
                    message="Invalid email or password"
                )
                
    except Exception as e:
        return LoginResponse(
            success=False,
            message=f"Login error: {str(e)}"
        )

@app.post("/auth/logout")
async def logout():
    """
    Logout endpoint - in production would invalidate JWT tokens
    """
    return {"success": True, "message": "Logged out successfully"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Dashboard metrics endpoint
@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    try:
        # Total consultants
        total_consultants = db.query(ConsultantProfile).count()
        
        # Bench consultants (available status)
        bench_consultants = db.query(ConsultantProfile).filter(
            ConsultantProfile.current_status == "available"
        ).count()
        
        # Ongoing projects (assigned opportunities)
        ongoing_projects = db.query(OpportunityApplication).filter(
            OpportunityApplication.status == "selected"
        ).count()
        
        # Open opportunities
        open_opportunities = db.query(ProjectOpportunity).filter(
            ProjectOpportunity.status == "open"
        ).count()
        
        # Active assignments
        active_assignments = db.query(OpportunityApplication).filter(
            OpportunityApplication.status.in_(["selected", "interview"])
        ).count()
        
        # Reports generated (resume analyses)
        reports_generated = db.query(ResumeAnalysis).count()
        
        return {
            "totalConsultants": total_consultants,
            "benchConsultants": bench_consultants,
            "ongoingProjects": ongoing_projects,
            "reportsGenerated": reports_generated,
            "activeAssignments": active_assignments,
            "openOpportunities": open_opportunities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Consultants list endpoint
@app.get("/api/consultants")
async def get_consultants(db: Session = Depends(get_db)):
    try:
        consultants = db.query(ConsultantProfile).join(User).all()
        result = []
        
        for consultant in consultants:
            # Get consultant skills
            skills = db.query(ConsultantSkill).filter(
                ConsultantSkill.consultant_id == consultant.id
            ).all()
            
            # Get active assignments
            assignments = db.query(OpportunityApplication).filter(
                and_(
                    OpportunityApplication.consultant_id == consultant.id,
                    OpportunityApplication.status == "selected"
                )
            ).all()
            
            consultant_data = {
                "id": str(consultant.id),
                "name": consultant.user.name,
                "email": consultant.user.email,
                "primary_skill": consultant.primary_skill,
                "skills": [skill.skill_name for skill in skills],
                "experience_years": consultant.experience_years,
                "department": consultant.user.department,
                "status": consultant.current_status,
                "attendance_rate": consultant.attendance_rate,
                "training_status": consultant.training_status,
                "resume_status": consultant.resume_status,
                "opportunities_count": consultant.opportunities_count,
                "bench_start_date": consultant.bench_start_date,
                "last_updated": consultant.updated_at.isoformat() if consultant.updated_at else None,
                "match_score": 0,  # Can be calculated based on requirements
                "active_assignments": [{"id": str(a.id), "status": a.status} for a in assignments],
                "assignment_count": len(assignments),
                "bench_status": consultant.current_status,
                "latest_resume": None  # Can be populated from resume analysis
            }
            result.append(consultant_data)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Individual consultant dashboard
@app.get("/api/consultant/dashboard/{email}")
async def get_consultant_dashboard(email: str, db: Session = Depends(get_db)):
    try:
        # Find consultant by email
        consultant = db.query(ConsultantProfile).join(User).filter(
            User.email == email
        ).first()
        
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        
        # Calculate training progress based on status
        training_progress_map = {
            "completed": 100.0,
            "in-progress": 50.0,
            "not-started": 0.0,
            "overdue": 25.0
        }
        training_progress = training_progress_map.get(consultant.training_status, 0.0)
        
        # Get consultant skills
        skills = db.query(ConsultantSkill).filter(
            ConsultantSkill.consultant_id == consultant.id
        ).all()
        
        # Get active assignments
        assignments = db.query(OpportunityApplication).filter(
            and_(
                OpportunityApplication.consultant_id == consultant.id,
                OpportunityApplication.status.in_(["selected", "interview"])
            )
        ).all()
        
        # Workflow steps
        workflow_steps = [
            {
                "id": "resume",
                "label": "Resume Updated",
                "completed": consultant.resume_status == "updated",
                "inProgress": consultant.resume_status == "pending"
            },
            {
                "id": "attendance",
                "label": "Attendance Reported", 
                "completed": consultant.attendance_rate > 80,
                "inProgress": False
            },
            {
                "id": "opportunities",
                "label": "Opportunities Documented",
                "completed": consultant.opportunities_count > 0,
                "inProgress": False
            },
            {
                "id": "training",
                "label": "Training Completed",
                "completed": consultant.training_status == "completed",
                "inProgress": consultant.training_status == "in-progress"
            }
        ]
        
        return {
            "consultant_name": consultant.user.name,
            "consultant_email": consultant.user.email,
            "resume_status": consultant.resume_status,
            "opportunities_count": consultant.opportunities_count,
            "attendance_rate": consultant.attendance_rate,
            "training_progress": training_progress,
            "training_status": consultant.training_status,
            "workflow_steps": workflow_steps,
            "skills": [{"name": s.skill_name, "level": s.proficiency_level} for s in skills],
            "match_score": 0,
            "last_resume_update": None,
            "active_assignments": [{"id": str(a.id), "status": a.status} for a in assignments],
            "assignment_count": len(assignments),
            "current_status": consultant.current_status
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Project opportunities endpoints
@app.get("/api/opportunities")
async def get_opportunities(db: Session = Depends(get_db)):
    try:
        opportunities = db.query(ProjectOpportunity).all()
        result = []
        
        for opp in opportunities:
            opportunity_data = {
                "id": str(opp.id),
                "title": opp.title,
                "description": opp.description,
                "client_name": opp.client_name,
                "required_skills": opp.required_skills or [],
                "experience_level": opp.experience_level,
                "project_duration": opp.project_duration,
                "budget_range": opp.budget_range,
                "start_date": str(opp.start_date) if opp.start_date else None,
                "end_date": str(opp.end_date) if opp.end_date else None,
                "status": opp.status,
                "created_at": str(opp.created_at) if opp.created_at else None
            }
            result.append(opportunity_data)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/opportunities")
async def create_opportunity(opportunity: OpportunityCreate, db: Session = Depends(get_db)):
    try:
        new_opportunity = ProjectOpportunity(
            title=opportunity.title,
            description=opportunity.description,
            client_name=opportunity.client_name,
            required_skills=opportunity.required_skills,
            experience_level=opportunity.experience_level,
            project_duration=opportunity.project_duration,
            budget_range=opportunity.budget_range,
            start_date=opportunity.start_date,  # Store as string
            end_date=opportunity.end_date,      # Store as string
            status="open"
        )
        
        db.add(new_opportunity)
        db.commit()
        db.refresh(new_opportunity)
        
        return {
            "id": str(new_opportunity.id),
            "title": new_opportunity.title,
            "status": "created",
            "message": "Opportunity created successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/opportunities/{opportunity_id}")
async def delete_opportunity(opportunity_id: int, db: Session = Depends(get_db)):
    try:
        opportunity = db.query(ProjectOpportunity).filter(
            ProjectOpportunity.id == opportunity_id
        ).first()
        
        if not opportunity:
            raise HTTPException(status_code=404, detail="Opportunity not found")
        
        db.delete(opportunity)
        db.commit()
        
        return {"message": "Opportunity deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.put("/api/opportunities/{opportunity_id}")
async def update_opportunity(
    opportunity_id: int, 
    opportunity: OpportunityUpdate, 
    db: Session = Depends(get_db)
):
    try:
        existing_opportunity = db.query(ProjectOpportunity).filter(
            ProjectOpportunity.id == opportunity_id
        ).first()
        
        if not existing_opportunity:
            raise HTTPException(status_code=404, detail="Opportunity not found")
        
        # Update fields
        if opportunity.title:
            existing_opportunity.title = opportunity.title
        if opportunity.description:
            existing_opportunity.description = opportunity.description
        if opportunity.status:
            existing_opportunity.status = opportunity.status
        
        existing_opportunity.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_opportunity)
        
        return {
            "id": str(existing_opportunity.id),
            "title": existing_opportunity.title,
            "status": "updated",
            "message": "Opportunity updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Attendance endpoints
@app.get("/api/attendance/{user_id}")
async def get_user_attendance(user_id: str, db: Session = Depends(get_db)):
    """Get attendance records for a specific user"""
    try:
        # Get last 30 days of attendance
        from datetime import date, timedelta
        start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        attendance_records = db.query(AttendanceRecord).filter(
            and_(
                AttendanceRecord.user_id == user_id,
                AttendanceRecord.date >= start_date
            )
        ).order_by(AttendanceRecord.date.desc()).all()
        
        result = []
        for record in attendance_records:
            result.append({
                "id": record.id,
                "date": record.date,
                "status": record.status,
                "check_in_time": record.check_in_time,
                "check_out_time": record.check_out_time,
                "hours_worked": record.hours_worked,
                "location": record.location,
                "notes": record.notes
            })
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching attendance: {str(e)}")

@app.post("/api/attendance")
async def mark_attendance(request: AttendanceRequest, db: Session = Depends(get_db)):
    """Mark attendance for a user"""
    try:
        # Check if attendance already exists for this date
        existing = db.query(AttendanceRecord).filter(
            and_(
                AttendanceRecord.user_id == request.user_id,
                AttendanceRecord.date == request.date
            )
        ).first()
        
        if existing:
            # Update existing record
            existing.status = request.status
            existing.check_in_time = request.check_in_time
            existing.check_out_time = request.check_out_time
            existing.hours_worked = request.hours_worked
            existing.location = request.location
            existing.notes = request.notes
            existing.updated_at = datetime.now()
        else:
            # Create new record
            attendance = AttendanceRecord(
                user_id=request.user_id,
                date=request.date,
                status=request.status,
                check_in_time=request.check_in_time,
                check_out_time=request.check_out_time,
                hours_worked=request.hours_worked,
                location=request.location,
                notes=request.notes
            )
            db.add(attendance)
        
        db.commit()
        return {"success": True, "message": "Attendance marked successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error marking attendance: {str(e)}")

@app.get("/api/attendance/summary/{user_id}")
async def get_attendance_summary(user_id: str, db: Session = Depends(get_db)):
    """Get attendance summary for a user"""
    try:
        from datetime import date, timedelta
        
        # Get last 30 days
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        # Get all records in date range
        records = db.query(AttendanceRecord).filter(
            and_(
                AttendanceRecord.user_id == user_id,
                AttendanceRecord.date >= start_date.strftime("%Y-%m-%d"),
                AttendanceRecord.date <= end_date.strftime("%Y-%m-%d")
            )
        ).all()
        
        # Calculate statistics
        total_days = len(records)
        present_days = len([r for r in records if r.status == 'present'])
        absent_days = len([r for r in records if r.status == 'absent'])
        half_days = len([r for r in records if r.status == 'half_day'])
        leave_days = len([r for r in records if r.status == 'leave'])
        
        total_hours = sum([r.hours_worked or 0 for r in records])
        attendance_rate = (present_days + (half_days * 0.5)) / max(total_days, 1) * 100
        
        return {
            "user_id": user_id,
            "period": f"{start_date} to {end_date}",
            "total_days": total_days,
            "present_days": present_days,
            "absent_days": absent_days,
            "half_days": half_days,
            "leave_days": leave_days,
            "total_hours": total_hours,
            "attendance_rate": round(attendance_rate, 1)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting attendance summary: {str(e)}")

# Resume upload and analysis endpoints
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), consultant_email: str = Form(...), db: Session = Depends(get_db)):
    """Upload and analyze resume, update consultant profile"""
    try:
        # Validate email parameter
        if not consultant_email or not consultant_email.strip():
            raise HTTPException(status_code=400, detail="Consultant email is required")
        
        # Import resume analyzer
        from resume_analyzer import get_resume_analyzer
        
        # Validate file type (temporarily allow .txt for testing)
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Create upload directory if it doesn't exist
        import os
        upload_dir = "uploads/resumes"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Read file content
        file_content = await file.read()
        
        # Save file to disk
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Get resume analyzer
        analyzer = get_resume_analyzer()
        
        # Analyze resume
        analysis_result = analyzer.analyze_resume(file_content, file.filename)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=400, detail=analysis_result["error"])
        
        # Find consultant by email if provided
        consultant = None
        if consultant_email:
            consultant = db.query(ConsultantProfile).join(User).filter(
                User.email == consultant_email
            ).first()
        
        # Save analysis to database
        if consultant:
            # Create resume analysis record
            resume_analysis = ResumeAnalysis(
                consultant_id=consultant.id,
                file_name=file.filename,
                file_path=f"uploads/resumes/{file.filename}",
                extracted_text=analysis_result["extracted_text"],
                extracted_skills=analysis_result["skills"],
                extracted_competencies=analysis_result["competencies"],
                identified_roles=analysis_result["roles"],
                skill_vector=analysis_result["skill_vector"],
                ai_summary=analysis_result["ai_summary"],
                ai_feedback=analysis_result["ai_feedback"],
                ai_suggestions=analysis_result["ai_suggestions"],
                confidence_score=analysis_result["confidence_score"],
                processing_time=2.5  # Mock processing time
            )
            db.add(resume_analysis)
            
            # Update consultant profile with new skills
            consultant.resume_status = "updated"
            consultant.updated_at = datetime.now()
            
            # Update primary skill based on most common skill
            if analysis_result["skills"]:
                consultant.primary_skill = analysis_result["skills"][0]
            
            # Clear existing skills and add new ones
            db.query(ConsultantSkill).filter(
                ConsultantSkill.consultant_id == consultant.id
            ).delete()
            
            # Add extracted skills
            for skill_name in analysis_result["skills"]:
                skill = ConsultantSkill(
                    consultant_id=consultant.id,
                    skill_name=skill_name,
                    proficiency_level="intermediate",  # Default level
                    years_experience=2.0,  # Default experience
                    source="resume",
                    confidence_score=0.8
                )
                db.add(skill)
            
            db.commit()
            
            # Create admin notification
            notification = AdminNotification(
                title="Resume Updated",
                message=f"{consultant.user.name} has uploaded a new resume with {len(analysis_result['skills'])} skills identified",
                notification_type="resume_upload",
                user_id=consultant.user_id,
                data={
                    "consultant_name": consultant.user.name,
                    "skills_count": len(analysis_result["skills"]),
                    "new_skills": analysis_result["skills"][:5]  # First 5 skills
                }
            )
            db.add(notification)
            db.commit()
        
        return {
            "success": True,
            "message": "Resume analyzed successfully",
            "analysis": analysis_result,
            "consultant_updated": consultant is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.get("/api/consultant/{consultant_id}/resume-analysis")
async def get_consultant_resume_analysis(consultant_id: int, db: Session = Depends(get_db)):
    """Get latest resume analysis for a consultant"""
    try:
        analysis = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.consultant_id == consultant_id
        ).order_by(ResumeAnalysis.created_at.desc()).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="No resume analysis found")
        
        return {
            "id": analysis.id,
            "filename": analysis.file_name,
            "skills": analysis.extracted_skills,
            "competencies": analysis.extracted_competencies,
            "roles": analysis.identified_roles,
            "ai_summary": analysis.ai_summary,
            "ai_feedback": analysis.ai_feedback,
            "ai_suggestions": analysis.ai_suggestions,
            "confidence_score": analysis.confidence_score,
            "created_at": analysis.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting resume analysis: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI Backend Server...")
    print("📊 Database: PostgreSQL consultant_bench_db")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("🔗 Frontend: http://localhost:3000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

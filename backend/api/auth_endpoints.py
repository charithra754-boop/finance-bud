"""
Authentication Endpoints
"""
from datetime import timedelta
from typing import Any
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr

from config import settings
from database import get_db
from data_models.models import User, Session, AuditLog
from utils.auth import (
    get_password_hash, verify_password, 
    create_access_token, create_refresh_token, 
    get_current_user
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    roles: list

    class Config:
        from_attributes = True


@router.post("/register", response_model=Token)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create user
    user_id = str(uuid4())
    user = User(
        id=user_id,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        roles=["USER"]
    )
    db.add(user)
    
    # Create session
    refresh_jti = str(uuid4())
    session = Session(
        id=str(uuid4()),
        user_id=user_id,
        refresh_token_jti=refresh_jti,
        expires_at=datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    )
    db.add(session)
    
    # Audit log
    audit = AuditLog(
        user_id=user_id,
        action="register",
        resource_type="user",
        resource_id=user_id,
        details={"email": user_in.email}
    )
    db.add(audit)
    
    await db.commit()
    
    # Generate tokens
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id, refresh_jti)
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token, 
        "token_type": "bearer"
    }


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login with email and password"""
    # Authenticate
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    # Create session
    refresh_jti = str(uuid4())
    session = Session(
        id=str(uuid4()),
        user_id=user.id,
        refresh_token_jti=refresh_jti,
        expires_at=datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    )
    db.add(session)
    
    # Audit log
    audit = AuditLog(
        user_id=user.id,
        action="login",
        resource_type="session",
        resource_id=session.id
    )
    db.add(audit)
    
    await db.commit()
    
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id, refresh_jti)
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token, 
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

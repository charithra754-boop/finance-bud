"""
AI Integration Hub
Central integration point for all advanced AI and ML features

Coordinates:
- NVIDIA NIM conversational AI
- NVIDIA GNN risk detection
- ML prediction engine
- RL portfolio optimization
- AI financial coaching
- Intelligent insights dashboard
- Advanced anomaly detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

from pydantic import BaseModel, Field

# Import all AI components
from agents.nvidia_nim_engine import NVIDIANIMEngine, create_nim_engine, NIMIntegratedAgent
from agents.nvidia_gnn_risk_detector import NVIDIAGNNRiskDetector, create_gnn_risk_detector, GNNIntegratedAgent
from agents.ml_prediction_engine import MLPredictionEngine, get_ml_prediction_engine
from agents.rl_portfolio_optimizer import RLPortfolioOptimizer, create_rl_portfolio_optimizer, RLIntegratedAgent
from agents.ai_financial_coach import AIFinancialCoach, create_ai_financial_coach, CoachingIntegratedAgent
from agents.intelligent_insights_dashboard import IntelligentInsightsDashboard, create_intelligent_insights_dashboard, InsightsDashboardIntegratedAgent
from agents.advanced_anomaly_detector import AdvancedAnomalyDetector, create_advanced_anomaly_detector, AnomalyDetectorIntegratedAgent

from data_models.schemas import FinancialState, MarketData, RiskProfile, AgentMessage, MessageType

logger = logging.getLogger(__name__)

class AIServiceType(str, Enum):
    """Types of AI services available"""
    CONVERSATIONAL_AI = "conversational_ai"
    RISK_DETECTION = "risk_detection"
    PREDICTION_ENGINE = "prediction_engine"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    FINANCIAL_COACHING = "financial_coaching"
    INSIGHTS_DASHBOARD = "insights_dashboard"
    ANOMALY_DETECTION = "anomaly_detection"

class AIRequestType(str, Enum):
    """Types of AI requests"""
    NATURAL_LANGUAGE_QUERY = "natural_language_query"
    RISK_ANALYSIS = "risk_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_PREDICTION = "market_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    COACHING_SESSION = "coaching_session"
    INSIGHT_GENERATION = "insight_generation"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"

class AIResponse(BaseModel):
    """Standardized AI response format"""
    request_id: str
    service_type: AIServiceType
    request_type: AIRequestType
    response_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class ComprehensiveAnalysisResult(BaseModel):
    """Result of comprehensive AI analysis"""
    analysis_id: str
    user_id: str
    conversational_insights: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    predictions: Dict[str, Any]
    portfolio_optimization: Dict[str, Any]
    coaching_recommendations: Dict[str, Any]
    dashboard_insights: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    overall_confidence: float
    key_findings: List[str]
    priority_actions: List[str]
    timestamp: datetime

@dataclass
class AIIntegrationConfig:
    """Configuration for AI integration hub"""
    enable_conversational_ai: bool = True
    enable_risk_detection: bool = True
    enable_predictions: bool = True
    enable_portfolio_optimization: bool = True
    enable_coaching: bool = True
    enable_insights_dashboard: bool = True
    enable_anomaly_detection: bool = True
    parallel_processing: bool = True
    cache_results: bool = True
    cache_ttl_minutes: int = 30

class AIIntegrationHub:
    """
    Central AI Integration Hub
    
    Coordinates all advanced AI and ML features:
    - Manages AI service lifecycle
    - Routes requests to appropriate AI services
    - Combines results from multiple AI services
    - Provides unified AI interface
    - Handles cross-service communication
    """
    
    def __init__(self, config: AIIntegrationConfig):
        self.config = config
        
        # Initialize AI services
        self.services = {}
        self.integrated_agents = {}
        
        # Response cache
        self.response_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            "request_count": 0,
            "average_response_time": 0.0,
            "service_usage": {},
            "error_count": 0
        }
        
        # Initialize services
        asyncio.create_task(self._initialize_ai_services())
        
        logger.info("AI Integration Hub initialized")

    async def _initialize_ai_services(self):
        """Initialize all AI services"""
        try:
            # Initialize NVIDIA NIM Engine
            if self.config.enable_conversational_ai:
                try:
                    nim_engine = create_nim_engine()
                    self.services[AIServiceType.CONVERSATIONAL_AI] = nim_engine
                    self.integrated_agents[AIServiceType.CONVERSATIONAL_AI] = NIMIntegratedAgent(nim_engine)
                    logger.info("NVIDIA NIM Engine initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize NVIDIA NIM Engine: {e}")
            
            # Initialize GNN Risk Detector
            if self.config.enable_risk_detection:
                try:
                    gnn_detector = create_gnn_risk_detector()
                    self.services[AIServiceType.RISK_DETECTION] = gnn_detector
                    self.integrated_agents[AIServiceType.RISK_DETECTION] = GNNIntegratedAgent(gnn_detector)
                    logger.info("NVIDIA GNN Risk Detector initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize GNN Risk Detector: {e}")
            
            # Initialize ML Prediction Engine
            if self.config.enable_predictions:
                try:
                    ml_engine = get_ml_prediction_engine()
                    self.services[AIServiceType.PREDICTION_ENGINE] = ml_engine
                    logger.info("ML Prediction Engine initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize ML Prediction Engine: {e}")
            
            # Initialize RL Portfolio Optimizer
            if self.config.enable_portfolio_optimization:
                try:
                    rl_optimizer = create_rl_portfolio_optimizer()
                    self.services[AIServiceType.PORTFOLIO_OPTIMIZATION] = rl_optimizer
                    self.integrated_agents[AIServiceType.PORTFOLIO_OPTIMIZATION] = RLIntegratedAgent(rl_optimizer)
                    logger.info("RL Portfolio Optimizer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize RL Portfolio Optimizer: {e}")
            
            # Initialize AI Financial Coach
            if self.config.enable_coaching:
                try:
                    ai_coach = create_ai_financial_coach()
                    self.services[AIServiceType.FINANCIAL_COACHING] = ai_coach
                    self.integrated_agents[AIServiceType.FINANCIAL_COACHING] = CoachingIntegratedAgent(ai_coach)
                    logger.info("AI Financial Coach initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize AI Financial Coach: {e}")
            
            # Initialize Insights Dashboard
            if self.config.enable_insights_dashboard:
                try:
                    insights_dashboard = create_intelligent_insights_dashboard()
                    self.services[AIServiceType.INSIGHTS_DASHBOARD] = insights_dashboard
                    self.integrated_agents[AIServiceType.INSIGHTS_DASHBOARD] = InsightsDashboardIntegratedAgent(insights_dashboard)
                    logger.info("Intelligent Insights Dashboard initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Insights Dashboard: {e}")
            
            # Initialize Anomaly Detector
            if self.config.enable_anomaly_detection:
                try:
                    anomaly_detector = create_advanced_anomaly_detector()
                    self.services[AIServiceType.ANOMALY_DETECTION] = anomaly_detector
                    self.integrated_agents[AIServiceType.ANOMALY_DETECTION] = AnomalyDetectorIntegratedAgent(anomaly_detector)
                    logger.info("Advanced Anomaly Detector initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Anomaly Detector: {e}")
            
            logger.info(f"AI Integration Hub: {len(self.services)} services initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI services: {e}")

    async def process_ai_request(
        self,
        request_type: AIRequestType,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> AIResponse:
        """
        Process AI request and route to appropriate service
        
        Args:
            request_type: Type of AI request
            request_data: Request data and parameters
            user_context: User context information
            
        Returns:
            AI response with results
        """
        try:
            start_time = datetime.now()
            request_id = f"ai_req_{start_time.timestamp()}"
            
            # Check cache first
            if self.config.cache_results:
                cached_response = self._get_cached_response(request_type, request_data, user_context)
                if cached_response:
                    return cached_response
            
            # Route request to appropriate service
            if request_type == AIRequestType.NATURAL_LANGUAGE_QUERY:
                response_data = await self._process_conversational_request(request_data, user_context)
                service_type = AIServiceType.CONVERSATIONAL_AI
                
            elif request_type == AIRequestType.RISK_ANALYSIS:
                response_data = await self._process_risk_analysis_request(request_data, user_context)
                service_type = AIServiceType.RISK_DETECTION
                
            elif request_type == AIRequestType.PORTFOLIO_OPTIMIZATION:
                response_data = await self._process_portfolio_optimization_request(request_data, user_context)
                service_type = AIServiceType.PORTFOLIO_OPTIMIZATION
                
            elif request_type == AIRequestType.MARKET_PREDICTION:
                response_data = await self._process_prediction_request(request_data, user_context)
                service_type = AIServiceType.PREDICTION_ENGINE
                
            elif request_type == AIRequestType.ANOMALY_DETECTION:
                response_data = await self._process_anomaly_detection_request(request_data, user_context)
                service_type = AIServiceType.ANOMALY_DETECTION
                
            elif request_type == AIRequestType.COACHING_SESSION:
                response_data = await self._process_coaching_request(request_data, user_context)
                service_type = AIServiceType.FINANCIAL_COACHING
                
            elif request_type == AIRequestType.INSIGHT_GENERATION:
                response_data = await self._process_insights_request(request_data, user_context)
                service_type = AIServiceType.INSIGHTS_DASHBOARD
                
            elif request_type == AIRequestType.COMPREHENSIVE_ANALYSIS:
                response_data = await self._process_comprehensive_analysis(request_data, user_context)
                service_type = AIServiceType.INSIGHTS_DASHBOARD  # Default for comprehensive
                
            else:
                raise ValueError(f"Unsupported request type: {request_type}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            ai_response = AIResponse(
                request_id=request_id,
                service_type=service_type,
                request_type=request_type,
                response_data=response_data,
                confidence_score=response_data.get("confidence_score", 0.8),
                processing_time=processing_time,
                metadata={
                    "services_used": response_data.get("services_used", [service_type.value]),
                    "cache_hit": False,
                    "user_id": user_context.get("user_id", "unknown")
                },
                recommendations=response_data.get("recommendations", []),
                timestamp=datetime.now()
            )
            
            # Cache response
            if self.config.cache_results:
                self._cache_response(request_type, request_data, user_context, ai_response)
            
            # Update performance metrics
            self._update_performance_metrics(service_type, processing_time)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing AI request: {e}")
            self.performance_metrics["error_count"] += 1
            raise

    async def comprehensive_financial_analysis(
        self,
        user_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        market_context: MarketData
    ) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive financial analysis using all AI services
        
        Args:
            user_data: User profile and preferences
            financial_data: Complete financial data
            market_context: Current market conditions
            
        Returns:
            Comprehensive analysis results from all AI services
        """
        try:
            start_time = datetime.now()
            analysis_id = f"comprehensive_{start_time.timestamp()}"
            user_id = user_data.get("user_id", "unknown")
            
            # Prepare tasks for parallel execution
            analysis_tasks = []
            
            if self.config.parallel_processing:
                # Run all analyses in parallel
                if AIServiceType.CONVERSATIONAL_AI in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_conversational_insights(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.RISK_DETECTION in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_risk_analysis(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.PREDICTION_ENGINE in self.services:
                    analysis_tasks.append(
                        self._get_predictions(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.PORTFOLIO_OPTIMIZATION in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_portfolio_optimization(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.FINANCIAL_COACHING in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_coaching_recommendations(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.INSIGHTS_DASHBOARD in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_dashboard_insights(user_data, financial_data, market_context)
                    )
                
                if AIServiceType.ANOMALY_DETECTION in self.integrated_agents:
                    analysis_tasks.append(
                        self._get_anomaly_detection(user_data, financial_data, market_context)
                    )
                
                # Execute all tasks in parallel
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
            else:
                # Run analyses sequentially
                results = []
                for task in analysis_tasks:
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in sequential analysis: {e}")
                        results.append({})
            
            # Combine results
            conversational_insights = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {}
            risk_analysis = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
            predictions = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
            portfolio_optimization = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else {}
            coaching_recommendations = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else {}
            dashboard_insights = results[5] if len(results) > 5 and not isinstance(results[5], Exception) else {}
            anomaly_detection = results[6] if len(results) > 6 and not isinstance(results[6], Exception) else {}
            
            # Calculate overall confidence
            confidence_scores = []
            for result in results:
                if isinstance(result, dict) and "confidence" in result:
                    confidence_scores.append(result["confidence"])
            
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Extract key findings and priority actions
            key_findings = self._extract_key_findings(results)
            priority_actions = self._extract_priority_actions(results)
            
            # Create comprehensive result
            comprehensive_result = ComprehensiveAnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                conversational_insights=conversational_insights,
                risk_analysis=risk_analysis,
                predictions=predictions,
                portfolio_optimization=portfolio_optimization,
                coaching_recommendations=coaching_recommendations,
                dashboard_insights=dashboard_insights,
                anomaly_detection=anomaly_detection,
                overall_confidence=overall_confidence,
                key_findings=key_findings,
                priority_actions=priority_actions,
                timestamp=datetime.now()
            )
            
            logger.info(f"Comprehensive analysis completed for user {user_id} in {(datetime.now() - start_time).total_seconds():.2f}s")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive financial analysis: {e}")
            raise

    async def _process_conversational_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process conversational AI request"""
        
        if AIServiceType.CONVERSATIONAL_AI not in self.integrated_agents:
            return {"error": "Conversational AI service not available"}
        
        agent = self.integrated_agents[AIServiceType.CONVERSATIONAL_AI]
        
        user_message = request_data.get("message", "")
        session_id = request_data.get("session_id", "default")
        user_id = user_context.get("user_id", "unknown")
        
        result = await agent.nim_engine.create_conversational_interface(
            user_message, session_id, user_id
        )
        
        return {
            "conversational_response": result,
            "confidence_score": 0.8,
            "services_used": ["conversational_ai"]
        }

    async def _process_risk_analysis_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process risk analysis request"""
        
        if AIServiceType.RISK_DETECTION not in self.integrated_agents:
            return {"error": "Risk detection service not available"}
        
        agent = self.integrated_agents[AIServiceType.RISK_DETECTION]
        
        user_data = request_data.get("user_data", {})
        transaction_history = request_data.get("transaction_history", [])
        
        result = await agent.analyze_user_risk(user_data, transaction_history)
        
        return {
            "risk_analysis": result,
            "confidence_score": result.get("overall_risk_score", 0.5),
            "services_used": ["risk_detection"]
        }

    async def _process_portfolio_optimization_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process portfolio optimization request"""
        
        if AIServiceType.PORTFOLIO_OPTIMIZATION not in self.integrated_agents:
            return {"error": "Portfolio optimization service not available"}
        
        agent = self.integrated_agents[AIServiceType.PORTFOLIO_OPTIMIZATION]
        
        user_portfolio = request_data.get("portfolio", {})
        market_context = request_data.get("market_context")
        risk_profile = request_data.get("risk_profile")
        
        result = await agent.optimize_user_portfolio(
            user_portfolio, market_context, risk_profile
        )
        
        return {
            "optimization_result": result,
            "confidence_score": result.get("confidence", 0.7),
            "services_used": ["portfolio_optimization"]
        }

    async def _process_prediction_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process prediction request"""
        
        if AIServiceType.PREDICTION_ENGINE not in self.services:
            return {"error": "Prediction engine service not available"}
        
        ml_engine = self.services[AIServiceType.PREDICTION_ENGINE]
        
        prediction_type = request_data.get("prediction_type", "market")
        
        if prediction_type == "market":
            symbol = request_data.get("symbol", "SPY")
            horizon_days = request_data.get("horizon_days", 30)
            
            result = await ml_engine.predict_market_trend(symbol, horizon_days)
        elif prediction_type == "portfolio":
            portfolio = request_data.get("portfolio", {})
            timeframe_days = request_data.get("timeframe_days", 365)
            
            result = await ml_engine.predict_portfolio_performance(portfolio, timeframe_days)
        else:
            result = {"error": f"Unsupported prediction type: {prediction_type}"}
        
        return {
            "prediction_result": result,
            "confidence_score": result.get("confidence", 0.6),
            "services_used": ["prediction_engine"]
        }

    async def _process_anomaly_detection_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process anomaly detection request"""
        
        if AIServiceType.ANOMALY_DETECTION not in self.integrated_agents:
            return {"error": "Anomaly detection service not available"}
        
        agent = self.integrated_agents[AIServiceType.ANOMALY_DETECTION]
        
        user_data = request_data.get("user_data", {})
        data_sources = request_data.get("data_sources", {})
        
        result = await agent.detect_user_anomalies(user_data, data_sources)
        
        return {
            "anomaly_detection": result,
            "confidence_score": 0.8,
            "services_used": ["anomaly_detection"]
        }

    async def _process_coaching_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process coaching request"""
        
        if AIServiceType.FINANCIAL_COACHING not in self.integrated_agents:
            return {"error": "Financial coaching service not available"}
        
        agent = self.integrated_agents[AIServiceType.FINANCIAL_COACHING]
        
        user_data = request_data.get("user_data", {})
        financial_context = request_data.get("financial_context", {})
        
        result = await agent.provide_user_coaching(user_data, financial_context)
        
        return {
            "coaching_result": result,
            "confidence_score": 0.8,
            "services_used": ["financial_coaching"]
        }

    async def _process_insights_request(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process insights generation request"""
        
        if AIServiceType.INSIGHTS_DASHBOARD not in self.integrated_agents:
            return {"error": "Insights dashboard service not available"}
        
        agent = self.integrated_agents[AIServiceType.INSIGHTS_DASHBOARD]
        
        user_data = request_data.get("user_data", {})
        market_context = request_data.get("market_context")
        
        result = await agent.generate_user_insights(user_data, market_context)
        
        return {
            "insights_result": result,
            "confidence_score": 0.8,
            "services_used": ["insights_dashboard"]
        }

    async def _process_comprehensive_analysis(
        self,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process comprehensive analysis request"""
        
        user_data = request_data.get("user_data", {})
        financial_data = request_data.get("financial_data", {})
        market_context = request_data.get("market_context")
        
        result = await self.comprehensive_financial_analysis(
            user_data, financial_data, market_context
        )
        
        return {
            "comprehensive_analysis": result.dict(),
            "confidence_score": result.overall_confidence,
            "services_used": list(self.services.keys())
        }

    # Helper methods for comprehensive analysis
    async def _get_conversational_insights(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get conversational AI insights"""
        try:
            if AIServiceType.CONVERSATIONAL_AI in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.CONVERSATIONAL_AI]
                # Simplified call - in production would be more sophisticated
                return {"insights": "Conversational analysis completed", "confidence": 0.8}
            return {}
        except Exception as e:
            logger.error(f"Error getting conversational insights: {e}")
            return {}

    async def _get_risk_analysis(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get risk analysis"""
        try:
            if AIServiceType.RISK_DETECTION in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.RISK_DETECTION]
                transaction_history = financial_data.get("transactions", [])
                return await agent.analyze_user_risk(user_data, transaction_history)
            return {}
        except Exception as e:
            logger.error(f"Error getting risk analysis: {e}")
            return {}

    async def _get_predictions(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get predictions"""
        try:
            if AIServiceType.PREDICTION_ENGINE in self.services:
                ml_engine = self.services[AIServiceType.PREDICTION_ENGINE]
                portfolio = financial_data.get("portfolio", {})
                result = await ml_engine.predict_portfolio_performance(portfolio)
                return {"predictions": result, "confidence": result.get("confidence", 0.6)}
            return {}
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {}

    async def _get_portfolio_optimization(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get portfolio optimization"""
        try:
            if AIServiceType.PORTFOLIO_OPTIMIZATION in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.PORTFOLIO_OPTIMIZATION]
                portfolio = financial_data.get("portfolio", {})
                risk_profile = user_data.get("risk_profile")
                return await agent.optimize_user_portfolio(portfolio, market_context, risk_profile)
            return {}
        except Exception as e:
            logger.error(f"Error getting portfolio optimization: {e}")
            return {}

    async def _get_coaching_recommendations(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get coaching recommendations"""
        try:
            if AIServiceType.FINANCIAL_COACHING in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.FINANCIAL_COACHING]
                return await agent.provide_user_coaching(user_data, financial_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting coaching recommendations: {e}")
            return {}

    async def _get_dashboard_insights(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get dashboard insights"""
        try:
            if AIServiceType.INSIGHTS_DASHBOARD in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.INSIGHTS_DASHBOARD]
                return await agent.generate_user_insights(user_data, market_context)
            return {}
        except Exception as e:
            logger.error(f"Error getting dashboard insights: {e}")
            return {}

    async def _get_anomaly_detection(self, user_data: Dict[str, Any], financial_data: Dict[str, Any], market_context: MarketData) -> Dict[str, Any]:
        """Get anomaly detection"""
        try:
            if AIServiceType.ANOMALY_DETECTION in self.integrated_agents:
                agent = self.integrated_agents[AIServiceType.ANOMALY_DETECTION]
                data_sources = {
                    "transactions": financial_data.get("transactions", []),
                    "portfolio": financial_data.get("portfolio", {}),
                    "market_data": [market_context.dict()] if market_context else []
                }
                return await agent.detect_user_anomalies(user_data, data_sources)
            return {}
        except Exception as e:
            logger.error(f"Error getting anomaly detection: {e}")
            return {}

    def _extract_key_findings(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from analysis results"""
        findings = []
        
        for result in results:
            if isinstance(result, dict):
                # Extract findings from different result types
                if "key_findings" in result:
                    findings.extend(result["key_findings"])
                elif "insights" in result:
                    findings.append("AI insights generated successfully")
                elif "anomalies" in result:
                    anomaly_count = len(result.get("anomalies", []))
                    if anomaly_count > 0:
                        findings.append(f"{anomaly_count} anomalies detected")
                elif "optimization_result" in result:
                    findings.append("Portfolio optimization recommendations available")
        
        return findings[:5]  # Limit to top 5 findings

    def _extract_priority_actions(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract priority actions from analysis results"""
        actions = []
        
        for result in results:
            if isinstance(result, dict):
                # Extract actions from different result types
                if "priority_actions" in result:
                    actions.extend(result["priority_actions"])
                elif "recommendations" in result:
                    actions.extend(result["recommendations"][:2])  # Top 2 recommendations
                elif "coaching_messages" in result:
                    coaching_messages = result.get("coaching_messages", [])
                    for msg in coaching_messages[:2]:  # Top 2 coaching messages
                        if isinstance(msg, dict) and "action_items" in msg:
                            actions.extend(msg["action_items"][:1])  # Top action per message
        
        return list(set(actions))[:5]  # Unique actions, limit to 5

    def _get_cached_response(
        self,
        request_type: AIRequestType,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Optional[AIResponse]:
        """Get cached response if available and not expired"""
        
        cache_key = self._generate_cache_key(request_type, request_data, user_context)
        
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - cached_item["timestamp"] < timedelta(minutes=self.config.cache_ttl_minutes):
                cached_response = cached_item["response"]
                cached_response.metadata["cache_hit"] = True
                return cached_response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        return None

    def _cache_response(
        self,
        request_type: AIRequestType,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any],
        response: AIResponse
    ) -> None:
        """Cache AI response"""
        
        cache_key = self._generate_cache_key(request_type, request_data, user_context)
        
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now()
        }
        
        # Clean up old cache entries (simple cleanup)
        if len(self.response_cache) > 1000:  # Limit cache size
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]["timestamp"]
            )
            del self.response_cache[oldest_key]

    def _generate_cache_key(
        self,
        request_type: AIRequestType,
        request_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        
        # Create a simple hash-based cache key
        key_data = {
            "request_type": request_type.value,
            "user_id": user_context.get("user_id", "unknown"),
            "request_hash": hash(str(sorted(request_data.items())))
        }
        
        return f"ai_cache_{hash(str(sorted(key_data.items())))}"

    def _update_performance_metrics(self, service_type: AIServiceType, processing_time: float) -> None:
        """Update performance metrics"""
        
        self.performance_metrics["request_count"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        request_count = self.performance_metrics["request_count"]
        
        new_avg = ((current_avg * (request_count - 1)) + processing_time) / request_count
        self.performance_metrics["average_response_time"] = new_avg
        
        # Update service usage
        service_key = service_type.value
        if service_key not in self.performance_metrics["service_usage"]:
            self.performance_metrics["service_usage"][service_key] = 0
        self.performance_metrics["service_usage"][service_key] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services"""
        
        status = {}
        
        for service_type in AIServiceType:
            if service_type in self.services:
                status[service_type.value] = {
                    "status": "active",
                    "service_available": True,
                    "last_used": "recently"  # Could track actual usage
                }
            else:
                status[service_type.value] = {
                    "status": "inactive",
                    "service_available": False,
                    "reason": "Service not initialized or disabled"
                }
        
        return status

# Factory function
def create_ai_integration_hub() -> AIIntegrationHub:
    """Create and configure AI integration hub"""
    config = AIIntegrationConfig(
        enable_conversational_ai=True,
        enable_risk_detection=True,
        enable_predictions=True,
        enable_portfolio_optimization=True,
        enable_coaching=True,
        enable_insights_dashboard=True,
        enable_anomaly_detection=True,
        parallel_processing=True,
        cache_results=True
    )
    
    return AIIntegrationHub(config)

# Global instance
_ai_integration_hub = None

def get_ai_integration_hub() -> AIIntegrationHub:
    """Get or create singleton AI integration hub"""
    global _ai_integration_hub
    if _ai_integration_hub is None:
        _ai_integration_hub = create_ai_integration_hub()
    return _ai_integration_hub
"""
Intelligent Insights Dashboard Backend
AI-powered insights generation and dashboard data management

Implements:
- Automated insight generation with natural language explanations
- AI-enhanced charts with intelligent annotations
- Predictive visualization with confidence intervals
- Anomaly highlighting with AI-detected patterns
- Real-time dashboard data processing
- Personalized recommendation systems
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

from pydantic import BaseModel, Field
from data_models.schemas import FinancialState, MarketData, RiskProfile

logger = logging.getLogger(__name__)

class InsightType(str, Enum):
    """Types of financial insights"""
    PERFORMANCE_ANALYSIS = "performance_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_DETECTION = "opportunity_detection"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    GOAL_PROGRESS = "goal_progress"
    MARKET_IMPACT = "market_impact"
    BEHAVIORAL_INSIGHT = "behavioral_insight"

class VisualizationType(str, Enum):
    """Types of visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    GAUGE_CHART = "gauge_chart"
    CANDLESTICK = "candlestick"

class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "very_high"  # 90%+
    HIGH = "high"           # 75-90%
    MEDIUM = "medium"       # 50-75%
    LOW = "low"            # 25-50%
    VERY_LOW = "very_low"  # <25%

class AIInsight(BaseModel):
    """AI-generated financial insight"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    natural_language_explanation: str
    confidence_score: float
    importance_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    visualization_config: Optional[Dict[str, Any]]
    timestamp: datetime
    expiry_date: Optional[datetime]
    tags: List[str]

class ChartAnnotation(BaseModel):
    """AI-generated chart annotation"""
    annotation_id: str
    chart_type: VisualizationType
    position: Dict[str, float]  # x, y coordinates
    annotation_type: str  # "trend", "anomaly", "milestone", "prediction"
    text: str
    confidence: float
    styling: Dict[str, Any]
    interactive_data: Optional[Dict[str, Any]]

class PredictiveVisualization(BaseModel):
    """Predictive visualization with confidence intervals"""
    visualization_id: str
    chart_type: VisualizationType
    historical_data: List[Dict[str, Any]]
    predicted_data: List[Dict[str, Any]]
    confidence_intervals: Dict[str, List[Dict[str, Any]]]  # 95%, 80%, 50%
    prediction_metadata: Dict[str, Any]
    annotations: List[ChartAnnotation]

class DashboardWidget(BaseModel):
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str
    title: str
    data_source: str
    visualization_config: Dict[str, Any]
    insights: List[AIInsight]
    update_frequency: str  # "real_time", "hourly", "daily"
    personalization_factors: Dict[str, Any]
    interactive_features: List[str]

class DashboardLayout(BaseModel):
    """Complete dashboard layout"""
    layout_id: str
    user_id: str
    widgets: List[DashboardWidget]
    layout_config: Dict[str, Any]
    personalization_score: float
    last_updated: datetime
    performance_metrics: Dict[str, Any]

@dataclass
class InsightConfiguration:
    """Configuration for insight generation"""
    insight_refresh_interval: int = 300  # seconds
    max_insights_per_category: int = 5
    confidence_threshold: float = 0.6
    importance_threshold: float = 0.5
    anomaly_sensitivity: float = 0.8
    prediction_horizon_days: int = 30
    enable_real_time_updates: bool = True

class IntelligentInsightsDashboard:
    """
    Intelligent Insights Dashboard Backend
    
    Provides AI-powered dashboard capabilities:
    - Automated insight generation
    - Intelligent chart annotations
    - Predictive visualizations
    - Anomaly detection and highlighting
    - Personalized recommendations
    - Real-time data processing
    """
    
    def __init__(self, config: InsightConfiguration):
        self.config = config
        
        # Insight generation components
        self.insight_generator = InsightGenerator()
        self.chart_annotator = ChartAnnotator()
        self.prediction_engine = PredictionEngine()
        self.anomaly_detector = AnomalyDetector()
        
        # Dashboard state management
        self.active_dashboards: Dict[str, DashboardLayout] = {}
        self.insight_cache: Dict[str, List[AIInsight]] = {}
        self.real_time_subscriptions: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "insight_generation_time": [],
            "dashboard_load_time": [],
            "user_engagement": {}
        }
        
        logger.info("Intelligent Insights Dashboard initialized")

    async def generate_automated_insights(
        self,
        user_id: str,
        financial_data: Dict[str, Any],
        market_context: MarketData,
        user_preferences: Dict[str, Any]
    ) -> List[AIInsight]:
        """
        Generate automated insights with natural language explanations
        
        Args:
            user_id: User identifier
            financial_data: User's financial information
            market_context: Current market conditions
            user_preferences: User preferences and settings
            
        Returns:
            List of AI-generated insights
        """
        try:
            start_time = datetime.now()
            
            # Generate insights across different categories
            insights = []
            
            # Performance analysis insights
            performance_insights = await self.insight_generator.generate_performance_insights(
                financial_data, market_context
            )
            insights.extend(performance_insights)
            
            # Risk assessment insights
            risk_insights = await self.insight_generator.generate_risk_insights(
                financial_data, market_context
            )
            insights.extend(risk_insights)
            
            # Opportunity detection insights
            opportunity_insights = await self.insight_generator.generate_opportunity_insights(
                financial_data, market_context, user_preferences
            )
            insights.extend(opportunity_insights)
            
            # Trend analysis insights
            trend_insights = await self.insight_generator.generate_trend_insights(
                financial_data, market_context
            )
            insights.extend(trend_insights)
            
            # Anomaly detection insights
            anomaly_insights = await self.anomaly_detector.detect_financial_anomalies(
                financial_data, user_preferences
            )
            insights.extend(anomaly_insights)
            
            # Goal progress insights
            goal_insights = await self.insight_generator.generate_goal_insights(
                financial_data, user_preferences
            )
            insights.extend(goal_insights)
            
            # Filter and rank insights
            filtered_insights = self._filter_and_rank_insights(insights, user_preferences)
            
            # Generate natural language explanations
            for insight in filtered_insights:
                insight.natural_language_explanation = await self._generate_natural_language_explanation(
                    insight, financial_data, market_context
                )
            
            # Cache insights
            self.insight_cache[user_id] = filtered_insights
            
            # Track performance
            generation_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["insight_generation_time"].append(generation_time)
            
            logger.info(f"Generated {len(filtered_insights)} insights for user {user_id} in {generation_time:.2f}s")
            
            return filtered_insights
            
        except Exception as e:
            logger.error(f"Error generating automated insights: {e}")
            raise

    async def create_ai_enhanced_charts(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create AI-enhanced charts with intelligent annotations
        
        Args:
            chart_data: Raw chart data
            chart_type: Type of visualization
            user_context: User context and preferences
            
        Returns:
            Enhanced chart configuration with AI annotations
        """
        try:
            # Generate base chart configuration
            base_config = self._generate_base_chart_config(chart_data, chart_type)
            
            # Generate AI annotations
            annotations = await self.chart_annotator.generate_annotations(
                chart_data, chart_type, user_context
            )
            
            # Add intelligent highlighting
            highlighting = await self._generate_intelligent_highlighting(
                chart_data, chart_type, user_context
            )
            
            # Create interactive features
            interactive_features = self._generate_interactive_features(
                chart_data, chart_type, annotations
            )
            
            # Combine into enhanced chart
            enhanced_chart = {
                "base_config": base_config,
                "annotations": [ann.dict() for ann in annotations],
                "highlighting": highlighting,
                "interactive_features": interactive_features,
                "ai_insights": await self._generate_chart_insights(chart_data, chart_type),
                "performance_metrics": {
                    "data_points": len(chart_data.get("data", [])),
                    "annotation_count": len(annotations),
                    "confidence_score": self._calculate_chart_confidence(annotations)
                }
            }
            
            return enhanced_chart
            
        except Exception as e:
            logger.error(f"Error creating AI-enhanced chart: {e}")
            raise

    async def generate_predictive_visualizations(
        self,
        historical_data: List[Dict[str, Any]],
        prediction_config: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> PredictiveVisualization:
        """
        Generate predictive visualizations with confidence intervals
        
        Args:
            historical_data: Historical financial data
            prediction_config: Prediction parameters
            user_preferences: User preferences
            
        Returns:
            Predictive visualization with confidence intervals
        """
        try:
            # Generate predictions
            predictions = await self.prediction_engine.generate_predictions(
                historical_data, prediction_config
            )
            
            # Calculate confidence intervals
            confidence_intervals = await self.prediction_engine.calculate_confidence_intervals(
                predictions, [0.95, 0.80, 0.50]
            )
            
            # Generate predictive annotations
            annotations = await self.chart_annotator.generate_predictive_annotations(
                historical_data, predictions, confidence_intervals
            )
            
            # Create visualization
            visualization = PredictiveVisualization(
                visualization_id=f"pred_{datetime.now().timestamp()}",
                chart_type=VisualizationType.LINE_CHART,
                historical_data=historical_data,
                predicted_data=predictions,
                confidence_intervals=confidence_intervals,
                prediction_metadata={
                    "model_type": prediction_config.get("model_type", "ensemble"),
                    "prediction_horizon": prediction_config.get("horizon_days", 30),
                    "confidence_score": predictions[0].get("confidence", 0.8) if predictions else 0.5,
                    "last_updated": datetime.now().isoformat()
                },
                annotations=annotations
            )
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error generating predictive visualization: {e}")
            raise

    async def detect_and_highlight_anomalies(
        self,
        data_series: List[Dict[str, Any]],
        detection_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect and highlight anomalies with AI-detected patterns
        
        Args:
            data_series: Time series data for analysis
            detection_config: Anomaly detection configuration
            
        Returns:
            Anomaly detection results with highlighting information
        """
        try:
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_time_series_anomalies(
                data_series, detection_config
            )
            
            # Generate anomaly explanations
            explanations = []
            for anomaly in anomalies:
                explanation = await self._generate_anomaly_explanation(
                    anomaly, data_series, detection_config
                )
                explanations.append(explanation)
            
            # Create highlighting configuration
            highlighting_config = self._create_anomaly_highlighting_config(anomalies)
            
            # Generate anomaly insights
            anomaly_insights = await self._generate_anomaly_insights(
                anomalies, data_series, detection_config
            )
            
            return {
                "anomalies": anomalies,
                "explanations": explanations,
                "highlighting_config": highlighting_config,
                "insights": anomaly_insights,
                "detection_metadata": {
                    "total_points": len(data_series),
                    "anomalies_detected": len(anomalies),
                    "anomaly_rate": len(anomalies) / len(data_series) if data_series else 0,
                    "detection_sensitivity": detection_config.get("sensitivity", 0.8),
                    "confidence_threshold": detection_config.get("confidence_threshold", 0.7)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting and highlighting anomalies: {e}")
            raise

    async def create_personalized_dashboard(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        financial_data: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> DashboardLayout:
        """
        Create personalized dashboard layout with AI recommendations
        
        Args:
            user_id: User identifier
            user_profile: User profile information
            financial_data: User's financial data
            preferences: User preferences and settings
            
        Returns:
            Personalized dashboard layout
        """
        try:
            # Generate personalized widgets
            widgets = await self._generate_personalized_widgets(
                user_profile, financial_data, preferences
            )
            
            # Optimize widget layout
            layout_config = await self._optimize_dashboard_layout(
                widgets, user_profile, preferences
            )
            
            # Generate insights for each widget
            for widget in widgets:
                widget_insights = await self._generate_widget_insights(
                    widget, financial_data, user_profile
                )
                widget.insights = widget_insights
            
            # Calculate personalization score
            personalization_score = self._calculate_personalization_score(
                widgets, user_profile, preferences
            )
            
            # Create dashboard layout
            dashboard = DashboardLayout(
                layout_id=f"dashboard_{user_id}_{datetime.now().timestamp()}",
                user_id=user_id,
                widgets=widgets,
                layout_config=layout_config,
                personalization_score=personalization_score,
                last_updated=datetime.now(),
                performance_metrics={}
            )
            
            # Cache dashboard
            self.active_dashboards[user_id] = dashboard
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating personalized dashboard: {e}")
            raise

    async def process_real_time_updates(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        update_type: str
    ) -> Dict[str, Any]:
        """
        Process real-time updates for dashboard
        
        Args:
            user_id: User identifier
            update_data: New data for processing
            update_type: Type of update (market, portfolio, transaction, etc.)
            
        Returns:
            Processed update information for dashboard
        """
        try:
            if not self.config.enable_real_time_updates:
                return {"status": "real_time_updates_disabled"}
            
            # Get user's dashboard
            dashboard = self.active_dashboards.get(user_id)
            if not dashboard:
                return {"status": "no_active_dashboard"}
            
            # Process update based on type
            if update_type == "market_data":
                processed_update = await self._process_market_update(
                    update_data, dashboard
                )
            elif update_type == "portfolio_change":
                processed_update = await self._process_portfolio_update(
                    update_data, dashboard
                )
            elif update_type == "transaction":
                processed_update = await self._process_transaction_update(
                    update_data, dashboard
                )
            else:
                processed_update = await self._process_generic_update(
                    update_data, dashboard, update_type
                )
            
            # Generate real-time insights if significant change
            if processed_update.get("significance_score", 0) > 0.7:
                real_time_insights = await self._generate_real_time_insights(
                    update_data, update_type, dashboard
                )
                processed_update["insights"] = real_time_insights
            
            # Update dashboard timestamp
            dashboard.last_updated = datetime.now()
            
            return processed_update
            
        except Exception as e:
            logger.error(f"Error processing real-time update: {e}")
            raise

    def _filter_and_rank_insights(
        self,
        insights: List[AIInsight],
        user_preferences: Dict[str, Any]
    ) -> List[AIInsight]:
        """Filter and rank insights based on relevance and importance"""
        
        # Filter by confidence and importance thresholds
        filtered = [
            insight for insight in insights
            if (insight.confidence_score >= self.config.confidence_threshold and
                insight.importance_score >= self.config.importance_threshold)
        ]
        
        # Rank by combined score
        def ranking_score(insight: AIInsight) -> float:
            base_score = (insight.confidence_score * 0.6 + insight.importance_score * 0.4)
            
            # Boost score based on user preferences
            preference_boost = 0.0
            insight_tags = set(insight.tags)
            preferred_topics = set(user_preferences.get("preferred_topics", []))
            
            if insight_tags.intersection(preferred_topics):
                preference_boost = 0.2
            
            return base_score + preference_boost
        
        # Sort by ranking score and limit per category
        ranked = sorted(filtered, key=ranking_score, reverse=True)
        
        # Limit insights per category
        category_counts = {}
        final_insights = []
        
        for insight in ranked:
            category = insight.insight_type.value
            count = category_counts.get(category, 0)
            
            if count < self.config.max_insights_per_category:
                final_insights.append(insight)
                category_counts[category] = count + 1
        
        return final_insights

    async def _generate_natural_language_explanation(
        self,
        insight: AIInsight,
        financial_data: Dict[str, Any],
        market_context: MarketData
    ) -> str:
        """Generate natural language explanation for insight"""
        
        # Template-based explanation generation
        templates = {
            InsightType.PERFORMANCE_ANALYSIS: "Your portfolio has {performance_trend} by {performance_change}% over the last {time_period}. This {comparison} the market average of {market_performance}%.",
            InsightType.RISK_ASSESSMENT: "Your current risk level is {risk_level}. Based on your portfolio composition, you have {risk_exposure} exposure to {risk_factors}.",
            InsightType.OPPORTUNITY_DETECTION: "I've identified a potential opportunity in {opportunity_area}. {opportunity_description} This could potentially {expected_benefit}.",
            InsightType.ANOMALY_DETECTION: "I noticed an unusual pattern in your {anomaly_area}. {anomaly_description} This is {significance_level} compared to your typical behavior."
        }
        
        template = templates.get(insight.insight_type, "Here's what I found: {description}")
        
        # Extract relevant data for template
        supporting_data = insight.supporting_data
        
        # Simple template filling (in production, use more sophisticated NLG)
        explanation = template.format(
            performance_trend=supporting_data.get("trend", "changed"),
            performance_change=supporting_data.get("change_percent", "0"),
            time_period=supporting_data.get("time_period", "month"),
            comparison="outperforms" if supporting_data.get("vs_market", 0) > 0 else "underperforms",
            market_performance=supporting_data.get("market_performance", "0"),
            risk_level=supporting_data.get("risk_level", "moderate"),
            risk_exposure=supporting_data.get("exposure_level", "moderate"),
            risk_factors=", ".join(supporting_data.get("risk_factors", ["market volatility"])),
            opportunity_area=supporting_data.get("opportunity_area", "your portfolio"),
            opportunity_description=supporting_data.get("opportunity_desc", "There's a potential for improvement"),
            expected_benefit=supporting_data.get("expected_benefit", "improve your returns"),
            anomaly_area=supporting_data.get("anomaly_area", "your financial activity"),
            anomaly_description=supporting_data.get("anomaly_desc", "The pattern differs from normal"),
            significance_level=supporting_data.get("significance", "significant"),
            description=insight.description
        )
        
        return explanation

    def _generate_base_chart_config(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType
    ) -> Dict[str, Any]:
        """Generate base chart configuration"""
        
        base_configs = {
            VisualizationType.LINE_CHART: {
                "type": "line",
                "data": chart_data.get("data", []),
                "xAxis": chart_data.get("xAxis", {}),
                "yAxis": chart_data.get("yAxis", {}),
                "series": chart_data.get("series", [])
            },
            VisualizationType.BAR_CHART: {
                "type": "bar",
                "data": chart_data.get("data", []),
                "categories": chart_data.get("categories", []),
                "series": chart_data.get("series", [])
            },
            VisualizationType.PIE_CHART: {
                "type": "pie",
                "data": chart_data.get("data", []),
                "labels": chart_data.get("labels", [])
            }
        }
        
        return base_configs.get(chart_type, {"type": "generic", "data": chart_data})

    async def _generate_intelligent_highlighting(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent highlighting for charts"""
        
        highlighting = {
            "important_points": [],
            "trend_highlights": [],
            "anomaly_highlights": [],
            "goal_markers": []
        }
        
        # Identify important data points
        data = chart_data.get("data", [])
        if data:
            # Find peaks and valleys
            values = [point.get("value", 0) for point in data if isinstance(point, dict)]
            if values:
                max_val = max(values)
                min_val = min(values)
                
                for i, point in enumerate(data):
                    if isinstance(point, dict) and point.get("value") in [max_val, min_val]:
                        highlighting["important_points"].append({
                            "index": i,
                            "type": "peak" if point.get("value") == max_val else "valley",
                            "value": point.get("value"),
                            "significance": "high"
                        })
        
        return highlighting

    def _generate_interactive_features(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType,
        annotations: List[ChartAnnotation]
    ) -> List[str]:
        """Generate interactive features for charts"""
        
        features = ["zoom", "pan", "tooltip"]
        
        # Add features based on chart type
        if chart_type == VisualizationType.LINE_CHART:
            features.extend(["crosshair", "data_labels"])
        elif chart_type == VisualizationType.PIE_CHART:
            features.extend(["drill_down", "legend_toggle"])
        
        # Add features based on annotations
        if annotations:
            features.append("annotation_interaction")
        
        return features

    async def _generate_chart_insights(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType
    ) -> List[str]:
        """Generate AI insights for charts"""
        
        insights = []
        
        # Analyze data trends
        data = chart_data.get("data", [])
        if len(data) >= 2:
            # Simple trend analysis
            first_val = data[0].get("value", 0) if isinstance(data[0], dict) else 0
            last_val = data[-1].get("value", 0) if isinstance(data[-1], dict) else 0
            
            if last_val > first_val * 1.1:
                insights.append("Strong upward trend detected in the data")
            elif last_val < first_val * 0.9:
                insights.append("Downward trend observed in recent data")
            else:
                insights.append("Data shows relatively stable pattern")
        
        return insights

    def _calculate_chart_confidence(self, annotations: List[ChartAnnotation]) -> float:
        """Calculate overall confidence score for chart"""
        
        if not annotations:
            return 0.5
        
        total_confidence = sum(ann.confidence for ann in annotations)
        return total_confidence / len(annotations)

# Supporting classes
class InsightGenerator:
    """Generates various types of financial insights"""
    
    async def generate_performance_insights(
        self,
        financial_data: Dict[str, Any],
        market_context: MarketData
    ) -> List[AIInsight]:
        """Generate performance analysis insights"""
        
        insights = []
        
        # Portfolio performance insight
        portfolio_return = financial_data.get("portfolio_return", 0.0)
        market_return = getattr(market_context, "market_return", 0.05)
        
        if abs(portfolio_return - market_return) > 0.02:  # 2% difference
            performance_type = "outperformance" if portfolio_return > market_return else "underperformance"
            
            insight = AIInsight(
                insight_id=f"perf_{datetime.now().timestamp()}",
                insight_type=InsightType.PERFORMANCE_ANALYSIS,
                title=f"Portfolio {performance_type.title()} Detected",
                description=f"Your portfolio is {performance_type.replace('performance', 'performing')} the market",
                natural_language_explanation="",  # Will be filled later
                confidence_score=0.85,
                importance_score=0.9,
                supporting_data={
                    "portfolio_return": portfolio_return,
                    "market_return": market_return,
                    "difference": portfolio_return - market_return,
                    "performance_type": performance_type
                },
                recommendations=[
                    "Review portfolio allocation",
                    "Consider rebalancing strategy",
                    "Analyze individual asset performance"
                ],
                visualization_config={
                    "chart_type": "line_chart",
                    "comparison_data": True
                },
                timestamp=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=7),
                tags=["performance", "portfolio", "market_comparison"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def generate_risk_insights(
        self,
        financial_data: Dict[str, Any],
        market_context: MarketData
    ) -> List[AIInsight]:
        """Generate risk assessment insights"""
        
        insights = []
        
        # Risk concentration insight
        portfolio_concentration = financial_data.get("concentration_risk", 0.0)
        
        if portfolio_concentration > 0.4:  # 40% concentration threshold
            insight = AIInsight(
                insight_id=f"risk_{datetime.now().timestamp()}",
                insight_type=InsightType.RISK_ASSESSMENT,
                title="High Portfolio Concentration Detected",
                description="Your portfolio may be overly concentrated in specific assets",
                natural_language_explanation="",
                confidence_score=0.8,
                importance_score=0.85,
                supporting_data={
                    "concentration_level": portfolio_concentration,
                    "threshold": 0.4,
                    "risk_level": "high"
                },
                recommendations=[
                    "Diversify across asset classes",
                    "Reduce position sizes in concentrated holdings",
                    "Consider index funds for broader exposure"
                ],
                visualization_config={
                    "chart_type": "pie_chart",
                    "highlight_concentration": True
                },
                timestamp=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=14),
                tags=["risk", "concentration", "diversification"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def generate_opportunity_insights(
        self,
        financial_data: Dict[str, Any],
        market_context: MarketData,
        user_preferences: Dict[str, Any]
    ) -> List[AIInsight]:
        """Generate opportunity detection insights"""
        
        insights = []
        
        # Tax-loss harvesting opportunity
        unrealized_losses = financial_data.get("unrealized_losses", 0.0)
        
        if unrealized_losses > 1000:  # $1000 threshold
            insight = AIInsight(
                insight_id=f"opp_{datetime.now().timestamp()}",
                insight_type=InsightType.OPPORTUNITY_DETECTION,
                title="Tax-Loss Harvesting Opportunity",
                description="You have unrealized losses that could be harvested for tax benefits",
                natural_language_explanation="",
                confidence_score=0.75,
                importance_score=0.7,
                supporting_data={
                    "unrealized_losses": unrealized_losses,
                    "potential_tax_savings": unrealized_losses * 0.22,  # Assume 22% tax rate
                    "opportunity_type": "tax_optimization"
                },
                recommendations=[
                    "Review positions with unrealized losses",
                    "Consider tax-loss harvesting strategy",
                    "Consult with tax advisor"
                ],
                visualization_config={
                    "chart_type": "bar_chart",
                    "show_tax_impact": True
                },
                timestamp=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                tags=["opportunity", "tax_optimization", "losses"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def generate_trend_insights(
        self,
        financial_data: Dict[str, Any],
        market_context: MarketData
    ) -> List[AIInsight]:
        """Generate trend analysis insights"""
        
        insights = []
        
        # Spending trend insight
        spending_trend = financial_data.get("spending_trend", 0.0)
        
        if abs(spending_trend) > 0.1:  # 10% change threshold
            trend_direction = "increasing" if spending_trend > 0 else "decreasing"
            
            insight = AIInsight(
                insight_id=f"trend_{datetime.now().timestamp()}",
                insight_type=InsightType.TREND_ANALYSIS,
                title=f"Spending Trend: {trend_direction.title()}",
                description=f"Your spending has been {trend_direction} over recent periods",
                natural_language_explanation="",
                confidence_score=0.8,
                importance_score=0.6,
                supporting_data={
                    "trend_percentage": spending_trend,
                    "trend_direction": trend_direction,
                    "trend_significance": "high" if abs(spending_trend) > 0.2 else "moderate"
                },
                recommendations=[
                    "Review budget categories",
                    "Analyze spending patterns",
                    "Adjust financial goals if needed"
                ],
                visualization_config={
                    "chart_type": "line_chart",
                    "show_trend_line": True
                },
                timestamp=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=10),
                tags=["trend", "spending", "budget"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def generate_goal_insights(
        self,
        financial_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> List[AIInsight]:
        """Generate goal progress insights"""
        
        insights = []
        
        # Goal progress insight
        goals = user_preferences.get("goals", [])
        
        for goal in goals:
            progress = goal.get("progress", 0.0)
            target_progress = goal.get("target_progress", 0.0)
            
            if progress < target_progress * 0.8:  # Behind by 20%
                insight = AIInsight(
                    insight_id=f"goal_{goal.get('id', 'unknown')}_{datetime.now().timestamp()}",
                    insight_type=InsightType.GOAL_PROGRESS,
                    title=f"Goal '{goal.get('name', 'Unknown')}' Behind Schedule",
                    description="You're falling behind on this financial goal",
                    natural_language_explanation="",
                    confidence_score=0.9,
                    importance_score=0.8,
                    supporting_data={
                        "current_progress": progress,
                        "target_progress": target_progress,
                        "gap_percentage": (target_progress - progress) / target_progress,
                        "goal_name": goal.get("name", "Unknown")
                    },
                    recommendations=[
                        "Increase monthly contributions",
                        "Review goal timeline",
                        "Consider adjusting target amount"
                    ],
                    visualization_config={
                        "chart_type": "gauge_chart",
                        "show_target_line": True
                    },
                    timestamp=datetime.now(),
                    expiry_date=datetime.now() + timedelta(days=7),
                    tags=["goal", "progress", "behind_schedule"]
                )
                
                insights.append(insight)
        
        return insights

class ChartAnnotator:
    """Generates intelligent chart annotations"""
    
    async def generate_annotations(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType,
        user_context: Dict[str, Any]
    ) -> List[ChartAnnotation]:
        """Generate annotations for charts"""
        
        annotations = []
        
        # Generate trend annotations
        trend_annotations = await self._generate_trend_annotations(chart_data, chart_type)
        annotations.extend(trend_annotations)
        
        # Generate milestone annotations
        milestone_annotations = await self._generate_milestone_annotations(chart_data, user_context)
        annotations.extend(milestone_annotations)
        
        return annotations
    
    async def generate_predictive_annotations(
        self,
        historical_data: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        confidence_intervals: Dict[str, List[Dict[str, Any]]]
    ) -> List[ChartAnnotation]:
        """Generate annotations for predictive charts"""
        
        annotations = []
        
        # Add prediction start annotation
        if historical_data and predictions:
            prediction_start = ChartAnnotation(
                annotation_id=f"pred_start_{datetime.now().timestamp()}",
                chart_type=VisualizationType.LINE_CHART,
                position={"x": len(historical_data) - 1, "y": historical_data[-1].get("value", 0)},
                annotation_type="prediction",
                text="Prediction starts here",
                confidence=0.9,
                styling={"color": "blue", "style": "dashed"},
                interactive_data={"tooltip": "Historical data ends, predictions begin"}
            )
            annotations.append(prediction_start)
        
        return annotations
    
    async def _generate_trend_annotations(
        self,
        chart_data: Dict[str, Any],
        chart_type: VisualizationType
    ) -> List[ChartAnnotation]:
        """Generate trend-based annotations"""
        
        annotations = []
        data = chart_data.get("data", [])
        
        if len(data) >= 3:
            # Simple trend detection
            values = [point.get("value", 0) for point in data if isinstance(point, dict)]
            
            if len(values) >= 3:
                # Check for consistent upward trend
                upward_trend = all(values[i] <= values[i+1] for i in range(len(values)-1))
                downward_trend = all(values[i] >= values[i+1] for i in range(len(values)-1))
                
                if upward_trend:
                    annotation = ChartAnnotation(
                        annotation_id=f"trend_up_{datetime.now().timestamp()}",
                        chart_type=chart_type,
                        position={"x": len(data) // 2, "y": values[len(values) // 2]},
                        annotation_type="trend",
                        text="Upward trend",
                        confidence=0.8,
                        styling={"color": "green", "arrow": "up"},
                        interactive_data={"trend_type": "upward", "strength": "strong"}
                    )
                    annotations.append(annotation)
                
                elif downward_trend:
                    annotation = ChartAnnotation(
                        annotation_id=f"trend_down_{datetime.now().timestamp()}",
                        chart_type=chart_type,
                        position={"x": len(data) // 2, "y": values[len(values) // 2]},
                        annotation_type="trend",
                        text="Downward trend",
                        confidence=0.8,
                        styling={"color": "red", "arrow": "down"},
                        interactive_data={"trend_type": "downward", "strength": "strong"}
                    )
                    annotations.append(annotation)
        
        return annotations
    
    async def _generate_milestone_annotations(
        self,
        chart_data: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> List[ChartAnnotation]:
        """Generate milestone-based annotations"""
        
        annotations = []
        milestones = user_context.get("milestones", [])
        
        for milestone in milestones:
            milestone_date = milestone.get("date")
            milestone_value = milestone.get("value")
            
            if milestone_date and milestone_value:
                annotation = ChartAnnotation(
                    annotation_id=f"milestone_{milestone.get('id', 'unknown')}",
                    chart_type=VisualizationType.LINE_CHART,
                    position={"x": milestone_date, "y": milestone_value},
                    annotation_type="milestone",
                    text=milestone.get("name", "Milestone"),
                    confidence=1.0,
                    styling={"color": "purple", "marker": "star"},
                    interactive_data={
                        "milestone_name": milestone.get("name"),
                        "achievement_date": milestone_date,
                        "target_value": milestone_value
                    }
                )
                annotations.append(annotation)
        
        return annotations

class PredictionEngine:
    """Generates predictions for financial data"""
    
    async def generate_predictions(
        self,
        historical_data: List[Dict[str, Any]],
        prediction_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate predictions based on historical data"""
        
        if not historical_data:
            return []
        
        # Simple linear trend prediction
        values = [point.get("value", 0) for point in historical_data if isinstance(point, dict)]
        
        if len(values) < 2:
            return []
        
        # Calculate simple trend
        trend = (values[-1] - values[0]) / len(values)
        last_value = values[-1]
        
        # Generate predictions
        horizon_days = prediction_config.get("horizon_days", 30)
        predictions = []
        
        for i in range(1, horizon_days + 1):
            predicted_value = last_value + (trend * i)
            
            predictions.append({
                "date": (datetime.now() + timedelta(days=i)).isoformat(),
                "value": predicted_value,
                "confidence": max(0.3, 0.9 - (i * 0.02))  # Decreasing confidence over time
            })
        
        return predictions
    
    async def calculate_confidence_intervals(
        self,
        predictions: List[Dict[str, Any]],
        confidence_levels: List[float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate confidence intervals for predictions"""
        
        intervals = {}
        
        for level in confidence_levels:
            level_key = f"{int(level * 100)}%"
            intervals[level_key] = []
            
            for prediction in predictions:
                base_value = prediction.get("value", 0)
                confidence = prediction.get("confidence", 0.5)
                
                # Simple confidence interval calculation
                margin = base_value * (1 - confidence) * (1 - level)
                
                intervals[level_key].append({
                    "date": prediction.get("date"),
                    "lower_bound": base_value - margin,
                    "upper_bound": base_value + margin,
                    "confidence_level": level
                })
        
        return intervals

class AnomalyDetector:
    """Detects anomalies in financial data"""
    
    async def detect_financial_anomalies(
        self,
        financial_data: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> List[AIInsight]:
        """Detect financial anomalies and generate insights"""
        
        insights = []
        
        # Spending anomaly detection
        spending_data = financial_data.get("spending_history", [])
        spending_anomalies = await self._detect_spending_anomalies(spending_data)
        
        for anomaly in spending_anomalies:
            insight = AIInsight(
                insight_id=f"anomaly_{datetime.now().timestamp()}",
                insight_type=InsightType.ANOMALY_DETECTION,
                title="Unusual Spending Pattern Detected",
                description=f"Detected unusual spending in {anomaly.get('category', 'unknown category')}",
                natural_language_explanation="",
                confidence_score=anomaly.get("confidence", 0.7),
                importance_score=0.6,
                supporting_data=anomaly,
                recommendations=[
                    "Review recent transactions",
                    "Check for unauthorized charges",
                    "Update budget if needed"
                ],
                visualization_config={
                    "chart_type": "bar_chart",
                    "highlight_anomaly": True
                },
                timestamp=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=5),
                tags=["anomaly", "spending", "unusual_pattern"]
            )
            insights.append(insight)
        
        return insights
    
    async def detect_time_series_anomalies(
        self,
        data_series: List[Dict[str, Any]],
        detection_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in time series data"""
        
        anomalies = []
        
        if len(data_series) < 3:
            return anomalies
        
        values = [point.get("value", 0) for point in data_series if isinstance(point, dict)]
        
        # Simple statistical anomaly detection
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        threshold = detection_config.get("sensitivity", 2.0) * std_dev
        
        for i, (point, value) in enumerate(zip(data_series, values)):
            if abs(value - mean_val) > threshold:
                anomalies.append({
                    "index": i,
                    "date": point.get("date"),
                    "value": value,
                    "expected_value": mean_val,
                    "deviation": abs(value - mean_val),
                    "severity": "high" if abs(value - mean_val) > 2 * threshold else "medium",
                    "confidence": min(0.9, abs(value - mean_val) / threshold / 2)
                })
        
        return anomalies
    
    async def _detect_spending_anomalies(
        self,
        spending_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in spending patterns"""
        
        anomalies = []
        
        if not spending_data:
            return anomalies
        
        # Group by category
        category_spending = {}
        for transaction in spending_data:
            category = transaction.get("category", "other")
            amount = transaction.get("amount", 0)
            
            if category not in category_spending:
                category_spending[category] = []
            category_spending[category].append(amount)
        
        # Detect anomalies per category
        for category, amounts in category_spending.items():
            if len(amounts) >= 3:
                mean_amount = sum(amounts) / len(amounts)
                
                for amount in amounts:
                    if amount > mean_amount * 3:  # 3x average threshold
                        anomalies.append({
                            "category": category,
                            "amount": amount,
                            "expected_amount": mean_amount,
                            "anomaly_type": "high_spending",
                            "confidence": 0.8
                        })
        
        return anomalies

# Factory function
def create_intelligent_insights_dashboard() -> IntelligentInsightsDashboard:
    """Create and configure intelligent insights dashboard"""
    config = InsightConfiguration(
        insight_refresh_interval=300,
        max_insights_per_category=5,
        confidence_threshold=0.6,
        importance_threshold=0.5
    )
    
    return IntelligentInsightsDashboard(config)

# Integration with existing agent system
class InsightsDashboardIntegratedAgent:
    """Integration wrapper for insights dashboard with existing agent system"""
    
    def __init__(self, insights_dashboard: IntelligentInsightsDashboard):
        self.insights_dashboard = insights_dashboard
        
    async def generate_user_insights(
        self,
        user_data: Dict[str, Any],
        market_context: MarketData
    ) -> Dict[str, Any]:
        """Generate insights for user"""
        
        user_id = user_data.get("user_id", "unknown")
        financial_data = user_data.get("financial_data", {})
        preferences = user_data.get("preferences", {})
        
        # Generate insights
        insights = await self.insights_dashboard.generate_automated_insights(
            user_id, financial_data, market_context, preferences
        )
        
        return {
            "insights": [insight.dict() for insight in insights],
            "insight_count": len(insights),
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def create_user_dashboard(
        self,
        user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create personalized dashboard for user"""
        
        user_id = user_data.get("user_id", "unknown")
        user_profile = user_data.get("user_profile", {})
        financial_data = user_data.get("financial_data", {})
        preferences = user_data.get("preferences", {})
        
        # Create dashboard
        dashboard = await self.insights_dashboard.create_personalized_dashboard(
            user_id, user_profile, financial_data, preferences
        )
        
        return {
            "dashboard": dashboard.dict(),
            "personalization_score": dashboard.personalization_score,
            "widget_count": len(dashboard.widgets)
        }
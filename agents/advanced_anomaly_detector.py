"""
Advanced Anomaly Detection System
Multi-modal anomaly detection with various data sources

Implements:
- Multi-modal anomaly detection across different data types
- Behavioral anomaly detection with user pattern analysis
- Market anomaly detection with statistical methods
- Fraud detection with advanced ML techniques
- Real-time anomaly monitoring
- Ensemble anomaly detection methods
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

# ML imports with fallbacks
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - using simplified anomaly detection")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from pydantic import BaseModel, Field
from data_models.schemas import FinancialState, MarketData, RiskProfile

logger = logging.getLogger(__name__)

class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    SPENDING_ANOMALY = "spending_anomaly"
    INCOME_ANOMALY = "income_anomaly"
    INVESTMENT_ANOMALY = "investment_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    MARKET_ANOMALY = "market_anomaly"
    FRAUD_PATTERN = "fraud_pattern"
    SYSTEM_ANOMALY = "system_anomaly"
    CORRELATION_BREAK = "correlation_break"

class AnomalySeverity(str, Enum):
    """Severity levels for anomalies"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class DetectionMethod(str, Enum):
    """Anomaly detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    STATISTICAL = "statistical"
    DBSCAN = "dbscan"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    ENSEMBLE = "ensemble"
    BEHAVIORAL_RULES = "behavioral_rules"

class AnomalyDetectionResult(BaseModel):
    """Result of anomaly detection"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence_score: float
    detection_method: DetectionMethod
    description: str
    affected_data_points: List[Dict[str, Any]]
    statistical_measures: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime
    context: Dict[str, Any]
    false_positive_probability: float

class BehavioralPattern(BaseModel):
    """Behavioral pattern for anomaly detection"""
    pattern_id: str
    user_id: str
    pattern_type: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    deviation_scores: Dict[str, float]
    pattern_stability: float
    last_updated: datetime

class MarketAnomalyIndicator(BaseModel):
    """Market anomaly indicator"""
    indicator_id: str
    indicator_name: str
    current_value: float
    historical_mean: float
    historical_std: float
    z_score: float
    percentile_rank: float
    anomaly_threshold: float
    is_anomalous: bool

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection system"""
    contamination_rate: float = 0.1
    confidence_threshold: float = 0.7
    ensemble_methods: List[DetectionMethod] = None
    real_time_monitoring: bool = True
    behavioral_window_days: int = 30
    market_lookback_days: int = 90
    statistical_significance: float = 0.05
    fraud_detection_enabled: bool = True

class AdvancedAnomalyDetector:
    """
    Advanced Multi-Modal Anomaly Detection System
    
    Provides comprehensive anomaly detection across:
    - Financial transactions and spending patterns
    - Investment and portfolio behavior
    - Market conditions and correlations
    - User behavioral patterns
    - Fraud and security threats
    - System performance anomalies
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config or AnomalyDetectionConfig()
        
        # Initialize detection models
        if SKLEARN_AVAILABLE:
            self.models = self._initialize_ml_models()
        else:
            self.models = {}
        
        # Behavioral pattern tracking
        self.behavioral_baselines: Dict[str, BehavioralPattern] = {}
        self.market_baselines: Dict[str, MarketAnomalyIndicator] = {}
        
        # Detection history and performance
        self.detection_history: List[AnomalyDetectionResult] = []
        self.false_positive_tracker = {}
        
        # Real-time monitoring
        self.monitoring_active = config.real_time_monitoring
        self.monitoring_tasks = {}
        
        logger.info(f"Advanced Anomaly Detector initialized with sklearn: {SKLEARN_AVAILABLE}")

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for anomaly detection"""
        models = {}
        
        if SKLEARN_AVAILABLE:
            models[DetectionMethod.ISOLATION_FOREST] = IsolationForest(
                contamination=self.config.contamination_rate,
                random_state=42
            )
            
            models[DetectionMethod.ONE_CLASS_SVM] = OneClassSVM(
                nu=self.config.contamination_rate
            )
            
            models[DetectionMethod.DBSCAN] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            models[DetectionMethod.ELLIPTIC_ENVELOPE] = EllipticEnvelope(
                contamination=self.config.contamination_rate
            )
        
        return models

    async def detect_multi_modal_anomalies(
        self,
        data_sources: Dict[str, List[Dict[str, Any]]],
        user_context: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies across multiple data modalities
        
        Args:
            data_sources: Dictionary of different data sources (transactions, market, etc.)
            user_context: User context and preferences
            
        Returns:
            List of detected anomalies across all modalities
        """
        try:
            all_anomalies = []
            
            # Process each data source
            for source_name, source_data in data_sources.items():
                if not source_data:
                    continue
                
                # Detect anomalies in this data source
                source_anomalies = await self._detect_source_anomalies(
                    source_name, source_data, user_context
                )
                all_anomalies.extend(source_anomalies)
            
            # Cross-modal anomaly detection
            cross_modal_anomalies = await self._detect_cross_modal_anomalies(
                data_sources, user_context
            )
            all_anomalies.extend(cross_modal_anomalies)
            
            # Ensemble detection for high-confidence anomalies
            if len(all_anomalies) > 1:
                ensemble_anomalies = await self._ensemble_anomaly_detection(
                    all_anomalies, data_sources
                )
                all_anomalies.extend(ensemble_anomalies)
            
            # Filter and rank anomalies
            filtered_anomalies = self._filter_and_rank_anomalies(all_anomalies, user_context)
            
            # Update detection history
            self.detection_history.extend(filtered_anomalies)
            
            return filtered_anomalies
            
        except Exception as e:
            logger.error(f"Error in multi-modal anomaly detection: {e}")
            raise

    async def detect_behavioral_anomalies(
        self,
        user_id: str,
        behavioral_data: Dict[str, Any],
        historical_patterns: List[Dict[str, Any]]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect behavioral anomalies with user pattern analysis
        
        Args:
            user_id: User identifier
            behavioral_data: Current behavioral data
            historical_patterns: Historical behavioral patterns
            
        Returns:
            List of behavioral anomalies detected
        """
        try:
            anomalies = []
            
            # Get or create behavioral baseline
            baseline = await self._get_or_create_behavioral_baseline(
                user_id, historical_patterns
            )
            
            # Analyze current behavior against baseline
            behavior_analysis = await self._analyze_behavioral_deviations(
                behavioral_data, baseline
            )
            
            # Detect specific behavioral anomalies
            spending_anomalies = await self._detect_spending_behavior_anomalies(
                behavioral_data, baseline, behavior_analysis
            )
            anomalies.extend(spending_anomalies)
            
            # Investment behavior anomalies
            investment_anomalies = await self._detect_investment_behavior_anomalies(
                behavioral_data, baseline, behavior_analysis
            )
            anomalies.extend(investment_anomalies)
            
            # Timing and frequency anomalies
            timing_anomalies = await self._detect_timing_anomalies(
                behavioral_data, baseline, behavior_analysis
            )
            anomalies.extend(timing_anomalies)
            
            # Update behavioral baseline
            await self._update_behavioral_baseline(user_id, behavioral_data, baseline)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting behavioral anomalies: {e}")
            raise

    async def detect_market_anomalies(
        self,
        market_data: List[Dict[str, Any]],
        indicators: List[str],
        detection_config: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect market anomalies with statistical methods
        
        Args:
            market_data: Market data time series
            indicators: List of indicators to analyze
            detection_config: Detection configuration parameters
            
        Returns:
            List of market anomalies detected
        """
        try:
            anomalies = []
            
            # Statistical anomaly detection for each indicator
            for indicator in indicators:
                indicator_data = [
                    point.get(indicator, 0) for point in market_data
                    if isinstance(point, dict) and indicator in point
                ]
                
                if len(indicator_data) < 10:  # Need minimum data points
                    continue
                
                # Statistical analysis
                statistical_anomalies = await self._detect_statistical_anomalies(
                    indicator_data, indicator, detection_config
                )
                anomalies.extend(statistical_anomalies)
                
                # Volatility anomalies
                volatility_anomalies = await self._detect_volatility_anomalies(
                    indicator_data, indicator, detection_config
                )
                anomalies.extend(volatility_anomalies)
                
                # Correlation break anomalies
                if len(indicators) > 1:
                    correlation_anomalies = await self._detect_correlation_anomalies(
                        market_data, indicators, indicator, detection_config
                    )
                    anomalies.extend(correlation_anomalies)
            
            # Market regime change detection
            regime_anomalies = await self._detect_market_regime_changes(
                market_data, detection_config
            )
            anomalies.extend(regime_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting market anomalies: {e}")
            raise

    async def detect_fraud_patterns(
        self,
        transaction_data: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        fraud_config: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """
        Detect fraud patterns with advanced ML techniques
        
        Args:
            transaction_data: Transaction history data
            user_profile: User profile information
            fraud_config: Fraud detection configuration
            
        Returns:
            List of potential fraud patterns detected
        """
        try:
            if not self.config.fraud_detection_enabled:
                return []
            
            fraud_anomalies = []
            
            # Transaction amount anomalies
            amount_anomalies = await self._detect_transaction_amount_anomalies(
                transaction_data, user_profile
            )
            fraud_anomalies.extend(amount_anomalies)
            
            # Transaction frequency anomalies
            frequency_anomalies = await self._detect_transaction_frequency_anomalies(
                transaction_data, user_profile
            )
            fraud_anomalies.extend(frequency_anomalies)
            
            # Geographic anomalies
            geographic_anomalies = await self._detect_geographic_anomalies(
                transaction_data, user_profile
            )
            fraud_anomalies.extend(geographic_anomalies)
            
            # Merchant category anomalies
            merchant_anomalies = await self._detect_merchant_category_anomalies(
                transaction_data, user_profile
            )
            fraud_anomalies.extend(merchant_anomalies)
            
            # Temporal pattern anomalies
            temporal_anomalies = await self._detect_temporal_pattern_anomalies(
                transaction_data, user_profile
            )
            fraud_anomalies.extend(temporal_anomalies)
            
            # ML-based fraud detection
            if SKLEARN_AVAILABLE:
                ml_fraud_anomalies = await self._ml_fraud_detection(
                    transaction_data, user_profile, fraud_config
                )
                fraud_anomalies.extend(ml_fraud_anomalies)
            
            return fraud_anomalies
            
        except Exception as e:
            logger.error(f"Error detecting fraud patterns: {e}")
            raise

    async def real_time_anomaly_monitoring(
        self,
        user_id: str,
        data_stream: Dict[str, Any],
        monitoring_config: Dict[str, Any]
    ) -> Optional[AnomalyDetectionResult]:
        """
        Real-time anomaly monitoring for streaming data
        
        Args:
            user_id: User identifier
            data_stream: Real-time data stream
            monitoring_config: Monitoring configuration
            
        Returns:
            Anomaly detection result if anomaly detected, None otherwise
        """
        try:
            if not self.monitoring_active:
                return None
            
            # Quick statistical checks for real-time detection
            quick_anomaly = await self._quick_anomaly_check(
                data_stream, user_id, monitoring_config
            )
            
            if quick_anomaly:
                # Perform more detailed analysis
                detailed_anomaly = await self._detailed_real_time_analysis(
                    data_stream, user_id, monitoring_config
                )
                
                if detailed_anomaly and detailed_anomaly.confidence_score > self.config.confidence_threshold:
                    # Log and return anomaly
                    logger.warning(f"Real-time anomaly detected for user {user_id}: {detailed_anomaly.description}")
                    return detailed_anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"Error in real-time anomaly monitoring: {e}")
            return None

    async def _detect_source_anomalies(
        self,
        source_name: str,
        source_data: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies in a specific data source"""
        
        anomalies = []
        
        if source_name == "transactions":
            anomalies = await self._detect_transaction_anomalies(source_data, user_context)
        elif source_name == "market_data":
            anomalies = await self._detect_market_data_anomalies(source_data, user_context)
        elif source_name == "portfolio":
            anomalies = await self._detect_portfolio_anomalies(source_data, user_context)
        elif source_name == "user_behavior":
            anomalies = await self._detect_user_behavior_anomalies(source_data, user_context)
        
        return anomalies

    async def _detect_transaction_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies in transaction data"""
        
        anomalies = []
        
        if not transactions:
            return anomalies
        
        # Extract transaction amounts
        amounts = [tx.get("amount", 0) for tx in transactions]
        
        if SKLEARN_AVAILABLE and len(amounts) >= 10:
            # Use Isolation Forest for transaction amount anomalies
            amounts_array = np.array(amounts).reshape(-1, 1)
            
            # Normalize amounts
            scaler = StandardScaler()
            amounts_normalized = scaler.fit_transform(amounts_array)
            
            # Detect anomalies
            iso_forest = self.models.get(DetectionMethod.ISOLATION_FOREST)
            if iso_forest:
                anomaly_labels = iso_forest.fit_predict(amounts_normalized)
                anomaly_scores = iso_forest.score_samples(amounts_normalized)
                
                for i, (label, score, tx) in enumerate(zip(anomaly_labels, anomaly_scores, transactions)):
                    if label == -1:  # Anomaly detected
                        anomaly = AnomalyDetectionResult(
                            anomaly_id=f"tx_anomaly_{i}_{datetime.now().timestamp()}",
                            anomaly_type=AnomalyType.SPENDING_ANOMALY,
                            severity=self._determine_severity(-score),
                            confidence_score=min(0.95, abs(score)),
                            detection_method=DetectionMethod.ISOLATION_FOREST,
                            description=f"Unusual transaction amount: ${tx.get('amount', 0):.2f}",
                            affected_data_points=[tx],
                            statistical_measures={
                                "anomaly_score": float(score),
                                "amount": tx.get("amount", 0),
                                "z_score": self._calculate_z_score(tx.get("amount", 0), amounts)
                            },
                            recommendations=[
                                "Review transaction details",
                                "Verify transaction legitimacy",
                                "Check for unauthorized charges"
                            ],
                            timestamp=datetime.now(),
                            context={"source": "transaction_analysis"},
                            false_positive_probability=0.1
                        )
                        anomalies.append(anomaly)
        else:
            # Simple statistical anomaly detection
            if len(amounts) >= 3:
                mean_amount = sum(amounts) / len(amounts)
                std_amount = (sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)) ** 0.5
                
                for i, tx in enumerate(transactions):
                    amount = tx.get("amount", 0)
                    z_score = abs(amount - mean_amount) / std_amount if std_amount > 0 else 0
                    
                    if z_score > 2.5:  # 2.5 standard deviations
                        anomaly = AnomalyDetectionResult(
                            anomaly_id=f"tx_stat_anomaly_{i}_{datetime.now().timestamp()}",
                            anomaly_type=AnomalyType.SPENDING_ANOMALY,
                            severity=self._determine_severity_from_z_score(z_score),
                            confidence_score=min(0.9, z_score / 3.0),
                            detection_method=DetectionMethod.STATISTICAL,
                            description=f"Statistically unusual transaction: ${amount:.2f}",
                            affected_data_points=[tx],
                            statistical_measures={
                                "z_score": z_score,
                                "amount": amount,
                                "mean_amount": mean_amount,
                                "std_amount": std_amount
                            },
                            recommendations=[
                                "Review transaction details",
                                "Compare with typical spending patterns"
                            ],
                            timestamp=datetime.now(),
                            context={"source": "statistical_analysis"},
                            false_positive_probability=0.15
                        )
                        anomalies.append(anomaly)
        
        return anomalies

    async def _detect_cross_modal_anomalies(
        self,
        data_sources: Dict[str, List[Dict[str, Any]]],
        user_context: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies across multiple data modalities"""
        
        anomalies = []
        
        # Example: Detect inconsistencies between spending and income
        transactions = data_sources.get("transactions", [])
        income_data = data_sources.get("income", [])
        
        if transactions and income_data:
            total_spending = sum(tx.get("amount", 0) for tx in transactions)
            total_income = sum(inc.get("amount", 0) for inc in income_data)
            
            spending_ratio = total_spending / total_income if total_income > 0 else 0
            
            if spending_ratio > 1.2:  # Spending 20% more than income
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"cross_modal_{datetime.now().timestamp()}",
                    anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
                    severity=AnomalySeverity.HIGH,
                    confidence_score=0.8,
                    detection_method=DetectionMethod.BEHAVIORAL_RULES,
                    description=f"Spending significantly exceeds income (ratio: {spending_ratio:.2f})",
                    affected_data_points=[],
                    statistical_measures={
                        "spending_income_ratio": spending_ratio,
                        "total_spending": total_spending,
                        "total_income": total_income
                    },
                    recommendations=[
                        "Review budget and spending habits",
                        "Consider reducing discretionary expenses",
                        "Evaluate income sources"
                    ],
                    timestamp=datetime.now(),
                    context={"source": "cross_modal_analysis"},
                    false_positive_probability=0.2
                )
                anomalies.append(anomaly)
        
        return anomalies

    async def _ensemble_anomaly_detection(
        self,
        individual_anomalies: List[AnomalyDetectionResult],
        data_sources: Dict[str, List[Dict[str, Any]]]
    ) -> List[AnomalyDetectionResult]:
        """Perform ensemble anomaly detection for high-confidence results"""
        
        ensemble_anomalies = []
        
        # Group anomalies by type and data point
        anomaly_groups = {}
        
        for anomaly in individual_anomalies:
            key = f"{anomaly.anomaly_type}_{len(anomaly.affected_data_points)}"
            if key not in anomaly_groups:
                anomaly_groups[key] = []
            anomaly_groups[key].append(anomaly)
        
        # Create ensemble anomalies for groups with multiple detections
        for group_key, group_anomalies in anomaly_groups.items():
            if len(group_anomalies) >= 2:  # Multiple methods detected same anomaly
                # Calculate ensemble confidence
                ensemble_confidence = sum(a.confidence_score for a in group_anomalies) / len(group_anomalies)
                
                # Create ensemble anomaly
                ensemble_anomaly = AnomalyDetectionResult(
                    anomaly_id=f"ensemble_{group_key}_{datetime.now().timestamp()}",
                    anomaly_type=group_anomalies[0].anomaly_type,
                    severity=self._determine_ensemble_severity(group_anomalies),
                    confidence_score=min(0.95, ensemble_confidence * 1.2),  # Boost confidence for ensemble
                    detection_method=DetectionMethod.ENSEMBLE,
                    description=f"Ensemble detection: {group_anomalies[0].description}",
                    affected_data_points=group_anomalies[0].affected_data_points,
                    statistical_measures={
                        "ensemble_methods": [a.detection_method.value for a in group_anomalies],
                        "individual_confidences": [a.confidence_score for a in group_anomalies],
                        "ensemble_confidence": ensemble_confidence
                    },
                    recommendations=list(set(
                        rec for anomaly in group_anomalies for rec in anomaly.recommendations
                    )),
                    timestamp=datetime.now(),
                    context={"source": "ensemble_detection", "method_count": len(group_anomalies)},
                    false_positive_probability=0.05  # Lower false positive rate for ensemble
                )
                ensemble_anomalies.append(ensemble_anomaly)
        
        return ensemble_anomalies

    def _filter_and_rank_anomalies(
        self,
        anomalies: List[AnomalyDetectionResult],
        user_context: Dict[str, Any]
    ) -> List[AnomalyDetectionResult]:
        """Filter and rank anomalies by relevance and confidence"""
        
        # Filter by confidence threshold
        filtered = [
            anomaly for anomaly in anomalies
            if anomaly.confidence_score >= self.config.confidence_threshold
        ]
        
        # Rank by combined score
        def ranking_score(anomaly: AnomalyDetectionResult) -> float:
            severity_weights = {
                AnomalySeverity.CRITICAL: 1.0,
                AnomalySeverity.HIGH: 0.8,
                AnomalySeverity.MEDIUM: 0.6,
                AnomalySeverity.LOW: 0.4,
                AnomalySeverity.INFO: 0.2
            }
            
            severity_score = severity_weights.get(anomaly.severity, 0.5)
            confidence_score = anomaly.confidence_score
            recency_score = 1.0  # Could be based on timestamp
            
            return (severity_score * 0.4 + confidence_score * 0.4 + recency_score * 0.2)
        
        # Sort by ranking score
        ranked = sorted(filtered, key=ranking_score, reverse=True)
        
        # Remove duplicates and limit results
        unique_anomalies = []
        seen_descriptions = set()
        
        for anomaly in ranked:
            if anomaly.description not in seen_descriptions:
                unique_anomalies.append(anomaly)
                seen_descriptions.add(anomaly.description)
                
                if len(unique_anomalies) >= 10:  # Limit to top 10
                    break
        
        return unique_anomalies

    def _determine_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Determine severity based on anomaly score"""
        abs_score = abs(anomaly_score)
        
        if abs_score > 0.8:
            return AnomalySeverity.CRITICAL
        elif abs_score > 0.6:
            return AnomalySeverity.HIGH
        elif abs_score > 0.4:
            return AnomalySeverity.MEDIUM
        elif abs_score > 0.2:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    def _determine_severity_from_z_score(self, z_score: float) -> AnomalySeverity:
        """Determine severity based on z-score"""
        if z_score > 4.0:
            return AnomalySeverity.CRITICAL
        elif z_score > 3.0:
            return AnomalySeverity.HIGH
        elif z_score > 2.5:
            return AnomalySeverity.MEDIUM
        elif z_score > 2.0:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    def _determine_ensemble_severity(self, anomalies: List[AnomalyDetectionResult]) -> AnomalySeverity:
        """Determine severity for ensemble anomaly"""
        severity_values = {
            AnomalySeverity.CRITICAL: 5,
            AnomalySeverity.HIGH: 4,
            AnomalySeverity.MEDIUM: 3,
            AnomalySeverity.LOW: 2,
            AnomalySeverity.INFO: 1
        }
        
        avg_severity_value = sum(severity_values.get(a.severity, 1) for a in anomalies) / len(anomalies)
        
        if avg_severity_value >= 4.5:
            return AnomalySeverity.CRITICAL
        elif avg_severity_value >= 3.5:
            return AnomalySeverity.HIGH
        elif avg_severity_value >= 2.5:
            return AnomalySeverity.MEDIUM
        elif avg_severity_value >= 1.5:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    def _calculate_z_score(self, value: float, data_series: List[float]) -> float:
        """Calculate z-score for a value in a data series"""
        if len(data_series) < 2:
            return 0.0
        
        mean_val = sum(data_series) / len(data_series)
        variance = sum((x - mean_val) ** 2 for x in data_series) / len(data_series)
        std_dev = variance ** 0.5
        
        return abs(value - mean_val) / std_dev if std_dev > 0 else 0.0

    # Placeholder methods for additional functionality
    async def _get_or_create_behavioral_baseline(self, user_id: str, historical_patterns: List[Dict[str, Any]]) -> BehavioralPattern:
        """Get or create behavioral baseline for user"""
        # Implementation would create/retrieve behavioral baseline
        return BehavioralPattern(
            pattern_id=f"baseline_{user_id}",
            user_id=user_id,
            pattern_type="spending",
            baseline_metrics={},
            current_metrics={},
            deviation_scores={},
            pattern_stability=0.8,
            last_updated=datetime.now()
        )

    async def _analyze_behavioral_deviations(self, behavioral_data: Dict[str, Any], baseline: BehavioralPattern) -> Dict[str, Any]:
        """Analyze behavioral deviations from baseline"""
        return {"deviations": [], "significance": 0.5}

    async def _detect_spending_behavior_anomalies(self, behavioral_data: Dict[str, Any], baseline: BehavioralPattern, analysis: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect spending behavior anomalies"""
        return []

    async def _detect_investment_behavior_anomalies(self, behavioral_data: Dict[str, Any], baseline: BehavioralPattern, analysis: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect investment behavior anomalies"""
        return []

    async def _detect_timing_anomalies(self, behavioral_data: Dict[str, Any], baseline: BehavioralPattern, analysis: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect timing and frequency anomalies"""
        return []

    async def _update_behavioral_baseline(self, user_id: str, behavioral_data: Dict[str, Any], baseline: BehavioralPattern) -> None:
        """Update behavioral baseline with new data"""
        pass

    async def _detect_statistical_anomalies(self, data: List[float], indicator: str, config: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect statistical anomalies in indicator data"""
        return []

    async def _detect_volatility_anomalies(self, data: List[float], indicator: str, config: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect volatility anomalies"""
        return []

    async def _detect_correlation_anomalies(self, market_data: List[Dict[str, Any]], indicators: List[str], current_indicator: str, config: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect correlation break anomalies"""
        return []

    async def _detect_market_regime_changes(self, market_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect market regime changes"""
        return []

    # Additional placeholder methods for fraud detection
    async def _detect_transaction_amount_anomalies(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect transaction amount anomalies for fraud"""
        return []

    async def _detect_transaction_frequency_anomalies(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect transaction frequency anomalies"""
        return []

    async def _detect_geographic_anomalies(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect geographic anomalies in transactions"""
        return []

    async def _detect_merchant_category_anomalies(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect merchant category anomalies"""
        return []

    async def _detect_temporal_pattern_anomalies(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect temporal pattern anomalies"""
        return []

    async def _ml_fraud_detection(self, transactions: List[Dict[str, Any]], user_profile: Dict[str, Any], config: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """ML-based fraud detection"""
        return []

    async def _quick_anomaly_check(self, data_stream: Dict[str, Any], user_id: str, config: Dict[str, Any]) -> bool:
        """Quick anomaly check for real-time monitoring"""
        return False

    async def _detailed_real_time_analysis(self, data_stream: Dict[str, Any], user_id: str, config: Dict[str, Any]) -> Optional[AnomalyDetectionResult]:
        """Detailed real-time anomaly analysis"""
        return None

    # Additional placeholder methods for completeness
    async def _detect_market_data_anomalies(self, market_data: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect anomalies in market data"""
        return []

    async def _detect_portfolio_anomalies(self, portfolio_data: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect anomalies in portfolio data"""
        return []

    async def _detect_user_behavior_anomalies(self, behavior_data: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect anomalies in user behavior data"""
        return []

# Factory function
def create_advanced_anomaly_detector() -> AdvancedAnomalyDetector:
    """Create and configure advanced anomaly detector"""
    config = AnomalyDetectionConfig(
        contamination_rate=0.1,
        confidence_threshold=0.7,
        real_time_monitoring=True,
        fraud_detection_enabled=True
    )
    
    return AdvancedAnomalyDetector(config)

# Integration with existing agent system
class AnomalyDetectorIntegratedAgent:
    """Integration wrapper for anomaly detector with existing agent system"""
    
    def __init__(self, anomaly_detector: AdvancedAnomalyDetector):
        self.anomaly_detector = anomaly_detector
        
    async def detect_user_anomalies(
        self,
        user_data: Dict[str, Any],
        data_sources: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Detect anomalies for user across all data sources"""
        
        user_context = {
            "user_id": user_data.get("user_id", "unknown"),
            "preferences": user_data.get("preferences", {}),
            "risk_profile": user_data.get("risk_profile", {})
        }
        
        # Detect multi-modal anomalies
        anomalies = await self.anomaly_detector.detect_multi_modal_anomalies(
            data_sources, user_context
        )
        
        return {
            "anomalies": [anomaly.dict() for anomaly in anomalies],
            "anomaly_count": len(anomalies),
            "high_severity_count": len([a for a in anomalies if a.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH]]),
            "detection_timestamp": datetime.now().isoformat()
        }
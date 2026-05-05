from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ContentCategory(str, Enum):
    """Broad, data-tailored categories derived from subject lines."""

    ACCOUNT_SECURITY_AUTH = "Account / Security / Auth"
    BILLING_PAYMENTS_INVOICES = "Billing / Payments / Invoices"
    ORDERS_SHIPPING_ECOM = "Orders / Shipping / E-commerce"
    SOCIAL_MEDIA_AND_FORUM_UPDATES = "Social Media Updates"
    LEADS_CONTACT_FORMS = "Leads / Contact Forms"
    APPOINTMENTS_BOOKINGS_EVENTS = "Appointments / Bookings / Events"
    NEWSLETTERS_MARKETING = "Newsletters / Marketing"
    SUPPORT_TICKETS_CS = "Support Tickets / Customer Service"
    LEGAL_POLICY_COMPLIANCE = "Legal / Policy / Compliance"
    DEVELOPER_TECH_ALERTS = "Developer / Ops Alerts"
    REAL_ESTATE_PROPERTY = "Real Estate / Property"
    EDU_COURSES_SCHOOL = "Education / Courses"
    HEALTHCARE_CLINIC = "Healthcare / Clinic"
    HR_JOBS_RECRUITING = "HR / Jobs / Recruiting"
    GENERAL_PERSONAL_PRIMARY = "General / Primary"


class EmailClassification(BaseModel):
    """Broad email classification result for structured outputs."""

    category: ContentCategory = Field(..., description="High-level category for the email")
    custom_labels: List[str] = Field(default_factory=list, description="Optional additional labels")

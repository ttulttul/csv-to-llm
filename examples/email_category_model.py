from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ContentCategory(str, Enum):
    """
    Enumeration for the primary content category of an email.
    Each category includes a description to guide an LLM's classification decision.
    """

    PRIMARY = "Primary"
    """For crucial, one-to-one conversations and messages that don't fit into other categories. Examples: Emails from family, friends, or direct messages from colleagues."""

    WORK = "Work"
    """For all job-related correspondence, project updates, team discussions, and internal communications. Examples: Project status reports, meeting invitations, client communications."""

    PROMOTIONS = "Promotions"
    """For all marketing messages, sales offers, discounts, and advertisements. Examples: Deals from online stores, product announcements, brand newsletters."""

    SOCIAL = "Social"
    """For notifications from social media platforms. Examples: LinkedIn connection requests, Facebook notifications, Twitter updates."""

    NEWSLETTERS = "Newsletters"
    """For subscribed content, blogs, and periodical updates that are not directly promotional. Examples: News digests, industry publications, Substack newsletters."""

    FINANCE = "Finance"
    """For financial statements, bills, invoices, payment confirmations, and receipts. Examples: Bank statements, credit card bills, online purchase receipts."""

    TRAVEL = "Travel"
    """For all travel-related confirmations and updates. Examples: Flight itineraries, hotel bookings, rental car confirmations, trip planning emails."""

    PERSONAL = "Personal"
    """For non-work, non-primary informal correspondence and interests. Examples: Hobby group discussions, personal appointments, event RSVPs."""

    ALERTS = "Alerts"
    """For automated notifications about system status, security, or account activity. Examples: Password reset alerts, package delivery notifications, account login warnings."""


class EmailCategory(BaseModel):
    """
    A Pydantic model to represent the categorization of an email message.
    Designed to assist an LLM by providing clear descriptions for each category.
    """

    content_category: ContentCategory = Field(
        ...,
        description="The primary category of the email, chosen from a set of predefined options with detailed descriptions."
    )

    action_required: bool = Field(
        default=False,
        description="Set to true if the email requires a specific action or follow-up from the user."
    )

    project: Optional[str] = Field(
        default=None,
        description="If the email is work-related, specify the project or workstream it belongs to."
    )

    custom_labels: List[str] = Field(
        default_factory=list,
        description="A list of custom, user-defined labels for more granular or personalized organization."
    )

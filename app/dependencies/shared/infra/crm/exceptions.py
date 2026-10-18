class CRMClientError(Exception):
    """
    Base exception class for CRM client errors.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ContactNotFoundError(CRMClientError):
    """
    Raised when a contact_id passed to a CRM method does not correspond
    to any existing contact. Applies to any method that takes contact_id
    as a parameter (update_contact, create_ticket, get_tickets).
    """
    def __init__(self, contact_id: str):
        super().__init__(f"No contact found with id: {contact_id}")
        self.contact_id = contact_id


class ContactAlreadyExistsError(CRMClientError):
    """
    Raised when create_contact is called with a phone_number that
    already belongs to an existing contact. phone_number is the sole
    identifier/unique key across the CRM, so duplicates are not allowed.
    """
    def __init__(self, phone_number: str):
        super().__init__(f"A contact already exists with phone_number: {phone_number}")
        self.phone_number = phone_number


class TicketNotFoundError(CRMClientError):
    """
    Raised when a ticket_id passed to a CRM method does not correspond
    to any existing ticket.
    """
    def __init__(self, ticket_id: str):
        super().__init__(f"No ticket found with id: {ticket_id}")
        self.ticket_id = ticket_id
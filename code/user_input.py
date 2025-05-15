from datetime import datetime
from utils.data_loader import load_historical_data
from utils.forecast import generate_forecast
from hotel import Hotel

def get_user_input():
    """Collects required information from the user"""
    print("=== Hotel Room Price Forecasting ===")
    hotel_id = input("Enter Hotel ID: ").strip()
    room_type = input("Enter Room Type (e.g., 'standard', 'deluxe'): ").strip().lower()
    check_in = input("Enter Check-in Date (YYYY-MM-DD): ").strip()
    check_out = input("Enter Check-out Date (YYYY-MM-DD): ").strip()
    return hotel_id, room_type, check_in, check_out

def validate_dates(check_in, check_out):
    """Validates and parses date inputs"""
    try:
        start_date = datetime.strptime(check_in, "%Y-%m-%d")
        end_date = datetime.strptime(check_out, "%Y-%m-%d")
        if start_date >= end_date:
            raise ValueError("Check-out date must be after check-in date")
        return start_date, end_date
    except ValueError as e:
        print(f"Invalid date format: {e}")
        return None, None

def main():
    # Get user input
    hotel_id, room_type, check_in, check_out = get_user_input()
    
    # Validate dates
    start_date, end_date = validate_dates(check_in, check_out)
    if not start_date or not end_date:
        return

    try:
        # Load historical data
        hotel = load_historical_data(hotel_id)
        if not hotel:
            raise ValueError(f"Hotel with ID {hotel_id} not found")
        
        # Find matching room type
        room = next((r for r in hotel.rooms if r.type.lower() == room_type), None)
        if not room:
            raise ValueError(f"Room type '{room_type}' not found in hotel {hotel_id}")
        
        # Generate forecast
        forecast_price = generate_forecast(
            historical_prices=room.price_history,
            start_date=start_date,
            end_date=end_date,
            room_type=room_type
        )
        
        # Display results
        print(f"\nForecasted price for {room_type} room from {check_in} to {check_out}:")
        print(f"${forecast_price:.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
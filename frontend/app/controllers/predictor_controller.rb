class PredictorController < ApplicationController
  def index
    @drivers = F1ApiClient.drivers
    @circuits = F1ApiClient.circuits
    @grid_positions = (1..20).to_a
  end

  def predict
    @drivers = F1ApiClient.drivers
    @circuits = F1ApiClient.circuits
    @grid_positions = (1..20).to_a

    if params[:driver].blank? || params[:circuit].blank? || params[:grid].blank?
      flash.now[:alert] = "Please select all options."
      render :index and return
    end

    @prediction = F1ApiClient.predict(
      driver: params[:driver],
      circuit: params[:circuit],
      grid: params[:grid].to_i
    )

    if @prediction.nil?
      flash.now[:alert] = "Failed to get prediction. Please ensure the API is running."
      render :index
    else
      render :predict
    end
  end
end

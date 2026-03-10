class AnalysisController < ApplicationController
  def index
    @stats = F1ApiClient.stats
    @single_race = F1ApiClient.single_race
    @confusion_matrix = F1ApiClient.confusion_matrix
  end
end

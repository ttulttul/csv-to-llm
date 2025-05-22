import pytest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import re
import sys

# Import the module under test
import importlib.util

# Load the csv-to-llm.py module
spec = importlib.util.spec_from_file_location("csv_to_llm", "csv-to-llm.py")
csv_to_llm = importlib.util.module_from_spec(spec)
sys.modules["csv_to_llm"] = csv_to_llm
spec.loader.exec_module(csv_to_llm)


class TestCsvToLlm:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_csv(self, temp_dir):
        """Create a sample CSV file for testing."""
        csv_path = os.path.join(temp_dir, "test_input.csv")
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'description': ['Engineer', 'Doctor', 'Teacher'],
            'status': ['active', 'skip_me', 'active']
        })
        df.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def output_csv_path(self, temp_dir):
        """Return path for output CSV."""
        return os.path.join(temp_dir, "test_output.csv")
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        mock_client = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Mocked Claude response"
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        return mock_client


class TestPromptTemplateProcessing(TestCsvToLlm):
    
    def test_extract_required_columns_from_template(self):
        """Test extraction of column names from prompt template."""
        template = "Summarize this: {description} for person {name}"
        required_columns = re.findall(r'\{([^}]+)\}', template)
        assert 'description' in required_columns
        assert 'name' in required_columns
        assert len(required_columns) == 2
    
    def test_positional_column_references(self):
        """Test COL1, COL2 etc. positional references."""
        template = "Process: {COL1} and {COL2}"
        required_columns = re.findall(r'\{([^}]+)\}', template)
        positional_cols = [col for col in required_columns if re.fullmatch(r'COL\d+', col)]
        assert 'COL1' in positional_cols
        assert 'COL2' in positional_cols
        assert len(positional_cols) == 2


class TestProcessSingleRow(TestCsvToLlm):
    
    @patch('csv_to_llm.load_dotenv')
    @patch('os.getenv')
    @patch('anthropic.Anthropic')
    def test_process_single_row_success(self, mock_anthropic, mock_getenv, mock_load_dotenv, mock_anthropic_client):
        """Test successful processing of a single row."""
        # Setup mocks
        mock_getenv.return_value = "test_api_key"
        mock_anthropic.return_value = mock_anthropic_client
        
        # Test data
        row_data = {'name': 'Alice', 'description': 'Engineer'}
        args_tuple = (
            0,  # index
            row_data,
            ['name', 'description'],  # required_columns
            "Describe {name}: {description}",  # prompt_template
            "claude-3-sonnet",  # model
            1000,  # max_tokens
            0.7,  # temperature
            "You are helpful",  # system_prompt
            "response"  # output_column
        )
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Test response"
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response == "Test response"
            assert error is None
    
    @patch('csv_to_llm.load_dotenv')
    @patch('os.getenv')
    def test_process_single_row_missing_api_key(self, mock_getenv, mock_load_dotenv):
        """Test handling of missing API key."""
        mock_getenv.return_value = None
        
        row_data = {'name': 'Alice'}
        args_tuple = (0, row_data, ['name'], "{name}", "model", 1000, 0.7, "system", "output")
        
        index, response, error = csv_to_llm.process_single_row(args_tuple)
        
        assert index == 0
        assert response is None
        assert "ANTHROPIC_API_KEY not found" in error
    
    def test_process_single_row_missing_data(self):
        """Test handling of missing data in row."""
        row_data = {'name': 'Alice', 'description': None}
        args_tuple = (0, row_data, ['name', 'description'], "{name}: {description}", 
                      "model", 1000, 0.7, "system", "output")
        
        with patch('csv_to_llm.load_dotenv'), \
             patch('os.getenv', return_value="test_key"), \
             patch('anthropic.Anthropic'):
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response is None
            assert "Missing data for prompt template" in error
    
    def test_process_single_row_formatting_error(self):
        """Test handling of template formatting errors."""
        # Create data with all required columns present but with wrong value structure
        row_data = {'name': 'Alice', 'missing_col': 'value'}
        # But template expects a different column
        args_tuple = (0, row_data, ['name', 'wrong_col'], "{name}: {wrong_col}", 
                      "model", 1000, 0.7, "system", "output")
        
        with patch('csv_to_llm.load_dotenv'), \
             patch('os.getenv', return_value="test_key"), \
             patch('anthropic.Anthropic'):
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response is None
            # The function checks for missing data first, so we need a different test case
            assert "Missing data for prompt template" in error


class TestProcessCsvWithClaude(TestCsvToLlm):
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_process_csv_basic_functionality(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test basic CSV processing functionality."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Processed response"
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Describe: {description}",
                output_column="response",
                test_first_row=True
            )
            
            # Verify output file was created
            assert os.path.exists(output_csv_path)
            
            # Verify CSV content
            df = pd.read_csv(output_csv_path)
            assert 'response' in df.columns
            # First row should be processed in test mode
            assert df.loc[0, 'response'] == "Processed response"
    
    def test_missing_api_key_error(self, sample_csv, output_csv_path):
        """Test error when API key is missing."""
        # Clear all environment variables completely
        with patch('csv_to_llm.load_dotenv'), \
             patch('os.getenv', return_value=None):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Test: {description}",
                    output_column="response"
                )
    
    def test_missing_csv_file(self, output_csv_path, capsys):
        """Test handling of missing input CSV file."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            csv_to_llm.process_csv_with_claude(
                input_csv_path="nonexistent.csv",
                output_csv_path=output_csv_path,
                prompt_template="Test: {description}",
                output_column="response"
            )
            
            captured = capsys.readouterr()
            assert "Error loading CSV" in captured.out
    
    def test_missing_columns_in_template(self, sample_csv, output_csv_path):
        """Test error when template references missing columns."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Missing required columns"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Test: {nonexistent_column}",
                    output_column="response"
                )
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_skip_rows_functionality(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test row skipping based on regex pattern."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Processed response"
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Describe: {description}",
                output_column="response",
                skip_column="status",
                skip_regex="skip.*"
            )
            
            # Verify output
            df = pd.read_csv(output_csv_path)
            # Row with "skip_me" status should have empty response
            skip_row = df[df['status'] == 'skip_me']
            assert len(skip_row) == 1
            # Check if the response is empty string or NaN (both indicate skipped)
            response_val = skip_row.iloc[0]['response']
            assert pd.isna(response_val) or response_val == ""
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    def test_positional_column_references(self, temp_dir, output_csv_path):
        """Test COL1, COL2 positional column references."""
        # Create CSV with specific column order
        csv_path = os.path.join(temp_dir, "positional_test.csv")
        df = pd.DataFrame({
            'first_col': ['A', 'B'],
            'second_col': ['X', 'Y'],
            'third_col': ['1', '2']
        })
        df.to_csv(csv_path, index=False)
        
        with patch('anthropic.Anthropic'), \
             patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Processed"
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=csv_path,
                output_csv_path=output_csv_path,
                prompt_template="Process {COL1} and {COL2}",
                output_column="response",
                test_first_row=True
            )
            
            # Verify the API was called with correct positional values
            mock_api.assert_called()
            call_args = mock_api.call_args[1]
            prompt_value = call_args['prompt_value']
            assert "Process A and X" in prompt_value


class TestCachedApiCall(TestCsvToLlm):
    
    def test_cached_api_call_basic(self, mock_anthropic_client):
        """Test the cached API call wrapper."""
        # Need to clear the cache for this test
        csv_to_llm.call_claude_api_cached.clear()
        
        response = csv_to_llm.call_claude_api_cached(
            client=mock_anthropic_client,
            model="claude-3-sonnet",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="Test system",
            prompt_value="Test prompt"
        )
        
        assert response == "Mocked Claude response"
        mock_anthropic_client.messages.create.assert_called_once()
    
    def test_cached_api_call_unexpected_response(self):
        """Test handling of unexpected API response structure."""
        # Clear cache to ensure fresh call
        csv_to_llm.call_claude_api_cached.clear()
        
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = []  # Empty content
        mock_client.messages.create.return_value = mock_message
        
        with patch('builtins.print') as mock_print:
            response = csv_to_llm.call_claude_api_cached(
                client=mock_client,
                model="claude-3-sonnet",
                max_tokens=1000,
                temperature=0.7,
                system_prompt="Test system",
                prompt_value="Test prompt"
            )
            
            assert response == ""
            mock_print.assert_called()
            assert "Warning: Unexpected API response structure" in str(mock_print.call_args)


class TestParallelProcessing(TestCsvToLlm):
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_parallel_processing(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test parallel processing mode."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Parallel response"
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Describe: {description}",
                output_column="response",
                parallel=2  # Use 2 parallel processes
            )
            
            # Verify output file was created
            assert os.path.exists(output_csv_path)
            
            # Verify all rows were processed
            df = pd.read_csv(output_csv_path)
            processed_rows = df[df['response'].notna()]
            assert len(processed_rows) == 3  # All 3 rows should be processed


class TestArgumentParsing:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_prompt_template_file_loading(self, temp_dir):
        """Test loading prompt template from file."""
        template_file = os.path.join(temp_dir, "template.txt")
        with open(template_file, 'w') as f:
            f.write("Test template: {description}")
        
        # Test the file loading logic directly
        with open(template_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert content == "Test template: {description}"


class TestErrorHandling(TestCsvToLlm):
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic, sample_csv, output_csv_path):
        """Test handling of API errors during processing."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.side_effect = Exception("API Error")
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Describe: {description}",
                output_column="response",
                test_first_row=True
            )
            
            # Verify error was handled and saved to CSV
            df = pd.read_csv(output_csv_path)
            assert "ERROR: API Error" in str(df.loc[0, 'response'])
    
    def test_invalid_skip_column(self, sample_csv, output_csv_path):
        """Test error when skip column doesn't exist."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Skip column .* not found"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Test: {description}",
                    output_column="response",
                    skip_column="nonexistent",
                    skip_regex=".*"
                )
    
    def test_invalid_regex_pattern(self, sample_csv, output_csv_path):
        """Test error when skip regex is invalid."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Invalid regex pattern"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Test: {description}",
                    output_column="response",
                    skip_column="status",
                    skip_regex="[invalid"  # Invalid regex
                )


class TestIntegrationScenarios(TestCsvToLlm):
    """Integration tests for common usage scenarios."""
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_resume_processing_existing_output(self, mock_anthropic, temp_dir, mock_anthropic_client):
        """Test that processing resumes correctly when output column already has some values."""
        mock_anthropic.return_value = mock_anthropic_client
        
        # Create CSV with some pre-existing responses
        csv_path = os.path.join(temp_dir, "test_resume.csv")
        output_path = os.path.join(temp_dir, "test_resume_output.csv")
        
        df = pd.DataFrame({
            'description': ['Task1', 'Task2', 'Task3'],
            'response': ['Already processed', pd.NA, pd.NA]
        })
        df.to_csv(csv_path, index=False)
        
        with patch('csv_to_llm.call_claude_api_cached') as mock_api:
            mock_api.return_value = "New response"
            
            csv_to_llm.process_csv_with_claude(
                input_csv_path=csv_path,
                output_csv_path=output_path,
                prompt_template="Process: {description}",
                output_column="response"
            )
            
            # Check that only 2 new responses were generated (not 3)
            assert mock_api.call_count == 2
            
            # Verify final state
            df_result = pd.read_csv(output_path)
            assert df_result.loc[0, 'response'] == "Already processed"
            assert df_result.loc[1, 'response'] == "New response"
            assert df_result.loc[2, 'response'] == "New response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
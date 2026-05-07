import pytest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import re
import sys
import json
from dataclasses import replace
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl

# Import the module under test
from csv_to_llm import core as csv_to_llm
from csv_to_llm import auto as csv_to_llm_auto
from csv_to_llm import cli as csv_to_llm_cli
from csv_to_llm.core import RowProcessingArgs
from csv_to_llm.auto import run_auto_mode, AutoModelDesign, AutoPlan, _escape_unknown_prompt_fields


@pytest.fixture(autouse=True)
def clear_llm_caches():
    """Keep file-backed LLM caches from leaking across tests."""

    cached_functions = [
        csv_to_llm.call_claude_api_cached,
        csv_to_llm.call_openai_api_cached,
        csv_to_llm.call_perplexity_api_cached,
        csv_to_llm._call_openai_structured_json_cached,
        csv_to_llm._call_openai_structured_iterative_json_cached,
        csv_to_llm._call_perplexity_structured_json_cached,
        csv_to_llm_auto._run_openai_auto_design_json_cached,
        csv_to_llm_auto._run_perplexity_auto_design_json_cached,
    ]
    for cached_function in cached_functions:
        cached_function.clear()
    csv_to_llm._get_pydantic_model_class.cache_clear()
    yield


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
    def sample_pydantic_model(self, temp_dir):
        """Create a temporary Pydantic model file for structured output tests."""
        model_path = os.path.join(temp_dir, "user_model.py")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(
                "from pydantic import BaseModel\n\n"
                "class EmailCategory(BaseModel):\n"
                "    category: str\n"
                "    explanation: str\n"
            )
        return model_path

    @pytest.fixture
    def nested_pydantic_model(self, temp_dir):
        """Create a temporary nested Pydantic model file for column flattening tests."""
        model_path = os.path.join(temp_dir, "nested_user_model.py")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(
                "from pydantic import BaseModel\n\n"
                "from enum import Enum\n\n"
                "class CostStructureType(str, Enum):\n"
                "    PAID_ADDON = 'paid_addon'\n\n"
                "class Pricing(BaseModel):\n"
                "    cost_structure: CostStructureType\n"
                "    custom_domain_support: bool\n\n"
                "class EmailDeliveryService(BaseModel):\n"
                "    provider_name: str\n"
                "    pricing_and_provisioning: Pricing\n"
                "    webmail_clients: list[str]\n"
            )
        return model_path

    @pytest.fixture
    def user_hierarchy_pydantic_model(self, temp_dir):
        """Create a nested User model for iterative structured output tests."""
        model_path = os.path.join(temp_dir, "user_hierarchy_model.py")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(
                "from pydantic import BaseModel, Field\n\n"
                "class Profile(BaseModel):\n"
                "    \"\"\"Biographical identity details for a user profile.\"\"\"\n"
                "    first_name: str = Field(description='Legal or commonly used given name')\n"
                "    last_name: str\n"
                "    age: int\n"
                "    bio: str | None = None\n\n"
                "class Address(BaseModel):\n"
                "    \"\"\"Postal mailing address for the user.\"\"\"\n"
                "    street: str\n"
                "    city: str\n"
                "    state: str\n"
                "    zip_code: str\n"
                "    country: str = 'USA'\n\n"
                "class AccountSettings(BaseModel):\n"
                "    is_premium: bool = False\n"
                "    receive_newsletter: bool = True\n"
                "    theme: str = 'dark'\n\n"
                "class User(BaseModel):\n"
                "    \"\"\"Complete account record assembled from source text.\"\"\"\n"
                "    id: int\n"
                "    profile: Profile\n"
                "    address: Address\n"
                "    settings: AccountSettings\n"
            )
        return model_path
    
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
    
    @patch('csv_to_llm.core.load_dotenv')
    @patch('os.getenv')
    @patch('anthropic.Anthropic')
    def test_process_single_row_success(self, mock_anthropic, mock_getenv, mock_load_dotenv, mock_anthropic_client):
        """Test successful processing of a single row."""
        # Setup mocks
        mock_getenv.return_value = "test_api_key"
        mock_anthropic.return_value = mock_anthropic_client
        
        # Test data
        row_data = {'name': 'Alice', 'description': 'Engineer'}
        args_tuple = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name', 'description'],
            prompt_template="Describe {name}: {description}",
            model="claude-3-sonnet",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="You are helpful",
            output_column="response",
            structured_config=None,
            max_retries=2,
            column_prefix=None,
        )
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Test response"
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response["output_value"] == "Test response"
            assert error is None
    
    @patch('csv_to_llm.core.load_dotenv')
    @patch('os.getenv')
    def test_process_single_row_missing_api_key(self, mock_getenv, mock_load_dotenv):
        """Test handling of missing API key."""
        mock_getenv.return_value = None
        
        row_data = {'name': 'Alice'}
        args_tuple = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name'],
            prompt_template="{name}",
            model="model",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="system",
            output_column="output",
            structured_config=None,
            max_retries=2,
            column_prefix=None,
        )
        
        index, response, error = csv_to_llm.process_single_row(args_tuple)
        
        assert index == 0
        assert response is None
        assert "ANTHROPIC_API_KEY not found" in error
    
    def test_process_single_row_blank_data_renders_empty_string(self):
        """Blank cells in prompt columns should not fail the row."""
        row_data = {'name': 'Alice', 'description': None}
        args_tuple = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name', 'description'],
            prompt_template="{name}: {description}",
            model="model",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="system",
            output_column="output",
            structured_config=None,
            max_retries=2,
            column_prefix=None,
        )
        
        with patch('csv_to_llm.core.load_dotenv'), \
             patch('os.getenv', return_value="test_key"), \
             patch('anthropic.Anthropic'), \
             patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
            mock_api.return_value = "Test response"
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response["output_value"] == "Test response"
            assert error is None
            assert mock_api.call_args.kwargs["prompt_value"] == "Alice: "
    
    def test_process_single_row_formatting_error(self):
        """Test handling of template formatting errors."""
        # Create data with all required columns present but with wrong value structure
        row_data = {'name': 'Alice', 'missing_col': 'value'}
        # But template expects a different column
        args_tuple = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name', 'wrong_col'],
            prompt_template="{name}: {wrong_col}",
            model="model",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="system",
            output_column="output",
            structured_config=None,
            max_retries=2,
            column_prefix=None,
        )
        
        with patch('csv_to_llm.core.load_dotenv'), \
             patch('os.getenv', return_value="test_key"), \
             patch('anthropic.Anthropic'):
            index, response, error = csv_to_llm.process_single_row(args_tuple)
            
            assert index == 0
            assert response is None
            assert "Formatting error" in error

    @patch('csv_to_llm.core.load_dotenv')
    @patch('os.getenv', return_value="test_key")
    @patch('anthropic.Anthropic')
    def test_process_single_row_retries_on_exception(self, mock_anthropic, mock_getenv, mock_load_dotenv, mock_anthropic_client):
        """Ensure retries fall back to uncached calls after a failure."""
        mock_anthropic.return_value = mock_anthropic_client
        row_data = {'name': 'Alice'}
        args = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name'],
            prompt_template="{name}",
            model="sonnet",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="system",
            output_column="output",
            structured_config=None,
            max_retries=2,
            column_prefix=None,
        )

        with patch('csv_to_llm.core.call_claude_api_cached', side_effect=Exception("boom")) as cached, \
             patch('csv_to_llm.core.call_claude_api_uncached', return_value="Recovered") as uncached:
            index, response, error = csv_to_llm.process_single_row(args)
            assert index == 0
            assert response["output_value"] == "Recovered"
            assert error is None
            cached.assert_called_once()
            uncached.assert_called_once()

    @patch('csv_to_llm.core.load_dotenv')
    @patch('os.getenv', return_value="test_key")
    @patch('anthropic.Anthropic')
    def test_process_single_row_retry_exceeds_limit(self, mock_anthropic, mock_getenv, mock_load_dotenv, mock_anthropic_client):
        """After exhausting retries the worker should return an error."""
        mock_anthropic.return_value = mock_anthropic_client
        row_data = {'name': 'Alice'}
        args = RowProcessingArgs(
            index=0,
            row_data=row_data,
            required_columns=['name'],
            prompt_template="{name}",
            model="sonnet",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="system",
            output_column="output",
            structured_config=None,
            max_retries=1,
            column_prefix=None,
        )

        with patch('csv_to_llm.core.call_claude_api_cached', side_effect=Exception("boom")), \
             patch('csv_to_llm.core.call_claude_api_uncached', side_effect=Exception("boom2")):
            index, response, error = csv_to_llm.process_single_row(args)
            assert index == 0
            assert response is None
            assert "LLM call failed" in error


class TestProcessCsvWithClaude(TestCsvToLlm):
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_process_csv_basic_functionality(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test basic CSV processing functionality."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
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
        with patch('csv_to_llm.core.load_dotenv'), \
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

    def test_prompt_template_without_placeholders(self, sample_csv, output_csv_path):
        """Test error when template lacks any placeholders."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Prompt template must contain at least one column identifier"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="This template has no placeholders at all",
                    output_column="response"
                )

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('anthropic.Anthropic')
    def test_skip_rows_functionality(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test row skipping based on regex pattern."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
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
             patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
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

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key', 'ANTHROPIC_API_KEY': ''})
    def test_structured_output_flow(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Ensure structured outputs invoke the OpenAI helper and populate the column."""
        with patch('csv_to_llm.core.call_openai_structured') as mock_structured:
            class Dummy(BaseModel):
                category: str = "Category"
                explanation: str = "Because"

            mock_structured.return_value = Dummy()

            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                model="gpt-4o-mini",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_field="category",
                test_first_row=True,
            )

            assert mock_structured.called
            df = pd.read_csv(output_csv_path)
            assert df.loc[0, 'response'] == "Category"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key', 'ANTHROPIC_API_KEY': ''})
    def test_structured_output_column_prefix(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Structured runs can populate prefixed columns for every field."""
        with patch('csv_to_llm.core.call_openai_structured') as mock_structured:
            class Dummy(BaseModel):
                category: str = "Category"
                explanation: str = "Because"

            mock_structured.return_value = Dummy()

            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                model="gpt-4o-mini",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_column_prefix="llm_",
                test_first_row=True,
            )

            df = pd.read_csv(output_csv_path)
            response_payload = json.loads(df.loc[0, 'response'])
            assert response_payload["category"] == "Category"
            assert response_payload["explanation"] == "Because"
            assert df.loc[0, 'llm_category'] == "Category"
            assert df.loc[0, 'llm_explanation'] == "Because"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key', 'ANTHROPIC_API_KEY': ''})
    def test_structured_output_column_prefix_flattens_nested_fields(self, sample_csv, output_csv_path, nested_pydantic_model):
        """Nested structured output objects should become individual prefixed columns."""
        with patch('csv_to_llm.core.call_openai_structured') as mock_structured:
            class CostStructureType(str, Enum):
                PAID_ADDON = "paid_addon"

            class Pricing(BaseModel):
                cost_structure: CostStructureType = CostStructureType.PAID_ADDON
                custom_domain_support: bool = True

            class Dummy(BaseModel):
                provider_name: str = "Shopify"
                pricing_and_provisioning: Pricing = Pricing()
                webmail_clients: list[str] = ["Roundcube"]

            mock_structured.return_value = Dummy()

            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                model="gpt-4o-mini",
                pydantic_model_path=nested_pydantic_model,
                pydantic_model_class="EmailDeliveryService",
                pydantic_model_column_prefix="llm_",
                test_first_row=True,
            )

            df = pd.read_csv(output_csv_path)
            assert df.loc[0, 'llm_provider_name'] == "Shopify"
            assert df.loc[0, 'llm_pricing_and_provisioning_cost_structure'] == "paid_addon"
            assert df.loc[0, 'llm_pricing_and_provisioning_custom_domain_support'] == True
            assert json.loads(df.loc[0, 'llm_webmail_clients']) == ["Roundcube"]
            assert 'llm_pricing_and_provisioning' not in df.columns

    def test_iterative_structured_output_fills_one_leaf_field_at_a_time(self, user_hierarchy_pydantic_model):
        """Iterative mode should query and reassemble nested models by leaf field."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=user_hierarchy_pydantic_model,
            model_class_name="User",
            output_field=None,
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            iterate_fields=True,
        )
        values = {
            "id": 42,
            "first_name": "Ada",
            "last_name": "Lovelace",
            "age": 36,
            "bio": None,
            "street": "1 Example St",
            "city": "London",
            "state": "London",
            "zip_code": "SW1A",
            "country": "UK",
            "is_premium": True,
            "receive_newsletter": False,
            "theme": "light",
        }

        mock_client = Mock()

        def parse_side_effect(**kwargs):
            model_cls = kwargs["text_format"]
            field_name = next(iter(model_cls.model_fields))
            response = Mock()
            response.output_parsed = model_cls(**{field_name: values[field_name]})
            return response

        mock_client.responses.parse.side_effect = parse_side_effect

        config = replace(config, iterate_parallelism=4)
        parsed = csv_to_llm.call_openai_structured_iterative(
            prompt_value="User record: Ada Lovelace",
            structured_config=config,
            openai_client=mock_client,
        )

        assert parsed.id == 42
        assert parsed.profile.first_name == "Ada"
        assert parsed.address.country == "UK"
        assert parsed.settings.theme == "light"
        assert mock_client.responses.parse.call_count == len(values)
        prompts = [
            call.kwargs["input"][1]["content"]
            for call in mock_client.responses.parse.call_args_list
        ]
        assert any("first_name (of type str) of this User Profile" in prompt for prompt in prompts)
        assert any("theme (of type str) of this User AccountSettings" in prompt for prompt in prompts)
        assert any("Complete account record assembled from source text." in prompt for prompt in prompts)
        assert any("Biographical identity details for a user profile." in prompt for prompt in prompts)
        assert any("Legal or commonly used given name" in prompt for prompt in prompts)

    def test_iterative_structured_output_uses_cache(self, user_hierarchy_pydantic_model):
        """Repeated iterative structured calls should reuse the full-model cache."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=user_hierarchy_pydantic_model,
            model_class_name="User",
            output_field=None,
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            iterate_fields=True,
        )
        values = {
            "id": 42,
            "first_name": "Ada",
            "last_name": "Lovelace",
            "age": 36,
            "bio": None,
            "street": "1 Main",
            "city": "London",
            "state": "London",
            "zip_code": "SW1A",
            "country": "UK",
            "is_premium": True,
            "receive_newsletter": False,
            "theme": "light",
        }
        mock_client = Mock()

        def parse_side_effect(**kwargs):
            model_cls = kwargs["text_format"]
            field_name = next(iter(model_cls.model_fields.keys()))
            response = Mock()
            response.output_parsed = model_cls(**{field_name: values[field_name]})
            return response

        mock_client.responses.parse.side_effect = parse_side_effect

        first = csv_to_llm.call_openai_structured_iterative("User record", config, mock_client)
        second = csv_to_llm.call_openai_structured_iterative("User record", config, mock_client)

        assert first.id == second.id == 42
        assert mock_client.responses.parse.call_count == len(values)

    def test_structured_output_websearch_passes_tools(self, sample_pydantic_model):
        """Normal structured output can enable OpenAI Responses web search."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=sample_pydantic_model,
            model_class_name="EmailCategory",
            output_field="category",
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            model_websearch=True,
        )
        mock_client = Mock()

        class Parsed(BaseModel):
            category: str
            explanation: str

        response = Mock()
        response.output_parsed = Parsed(category="Category", explanation="Because")
        mock_client.responses.parse.return_value = response

        csv_to_llm.call_openai_structured(
            prompt_value="Categorize this",
            structured_config=config,
            openai_client=mock_client,
        )

        assert mock_client.responses.parse.call_args.kwargs["tools"] == [{"type": "web_search"}]

    def test_openai_structured_output_uses_cache(self, sample_pydantic_model):
        """Repeated OpenAI structured calls should reuse the joblib cache."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=sample_pydantic_model,
            model_class_name="EmailCategory",
            output_field="category",
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
        )
        model_cls = csv_to_llm._get_pydantic_model_class(sample_pydantic_model, "EmailCategory")
        mock_client = Mock()
        response = Mock()
        response.output_parsed = model_cls(category="Category", explanation="Because")
        mock_client.responses.parse.return_value = response

        first = csv_to_llm.call_openai_structured("Categorize this", config, mock_client)
        second = csv_to_llm.call_openai_structured("Categorize this", config, mock_client)

        assert first.category == second.category == "Category"
        assert mock_client.responses.parse.call_count == 1

    def test_perplexity_structured_output_uses_json_schema(self, sample_pydantic_model):
        """Perplexity structured output should send JSON Schema and validate output_text."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=sample_pydantic_model,
            model_class_name="EmailCategory",
            output_field="category",
            llm_model="pro-search",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            provider="perplexity",
            model_websearch=True,
        )
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "category": "Category",
            "explanation": "Because",
        })
        mock_client.responses.create.return_value = mock_response

        parsed = csv_to_llm.call_perplexity_structured(
            prompt_value="Categorize this",
            structured_config=config,
            perplexity_client=mock_client,
        )

        assert parsed.category == "Category"
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["preset"] == "pro-search"
        assert kwargs["input"] == "Categorize this"
        assert kwargs["instructions"] == "system"
        assert kwargs["max_output_tokens"] == 1000
        assert kwargs["tools"] == [{"type": "web_search"}, {"type": "fetch_url"}]
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "EmailCategory"
        assert kwargs["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert kwargs["response_format"]["json_schema"]["schema"]["required"] == ["category", "explanation"]

    def test_perplexity_response_format_strips_unsupported_schema_formats(self):
        """Perplexity rejects Pydantic URL format annotations such as format=uri."""

        class ProviderHomepage(BaseModel):
            homepage_url: HttpUrl | None

        response_format = csv_to_llm._perplexity_response_format(ProviderHomepage, "ProviderHomepage")
        schema_text = json.dumps(response_format["json_schema"]["schema"])

        assert '"format"' not in schema_text
        assert "homepage_url" in response_format["json_schema"]["schema"]["properties"]

    def test_perplexity_response_format_requires_optional_fields(self):
        """Perplexity requires every property to appear in strict required arrays."""

        class ProviderHeadcount(BaseModel):
            provider_headcount_estimate: Optional[int] = None

        response_format = csv_to_llm._perplexity_response_format(
            ProviderHeadcount,
            "ProviderHeadcount",
        )
        schema = response_format["json_schema"]["schema"]
        schema_text = json.dumps(schema)

        assert schema["required"] == ["provider_headcount_estimate"]
        assert schema["additionalProperties"] is False
        assert '"default"' not in schema_text

    def test_perplexity_response_format_requires_nested_object_fields(self):
        """Nested object schemas should also satisfy Perplexity strict mode."""

        class HeadcountDetail(BaseModel):
            estimate: Optional[int] = None

        class ProviderHeadcount(BaseModel):
            detail: HeadcountDetail

        response_format = csv_to_llm._perplexity_response_format(
            ProviderHeadcount,
            "ProviderHeadcount",
        )
        schema = response_format["json_schema"]["schema"]
        nested_schema = schema["$defs"]["HeadcountDetail"]

        assert schema["required"] == ["detail"]
        assert nested_schema["required"] == ["estimate"]
        assert nested_schema["additionalProperties"] is False

    def test_perplexity_structured_output_uses_cache(self, sample_pydantic_model):
        """Repeated Perplexity structured calls should reuse the joblib cache."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=sample_pydantic_model,
            model_class_name="EmailCategory",
            output_field="category",
            llm_model="pro-search",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            provider="perplexity",
        )
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = json.dumps({"category": "Category", "explanation": "Because"})
        mock_client.responses.create.return_value = mock_response

        first = csv_to_llm.call_perplexity_structured("Categorize this", config, mock_client)
        second = csv_to_llm.call_perplexity_structured("Categorize this", config, mock_client)

        assert first.category == second.category == "Category"
        assert mock_client.responses.create.call_count == 1

    def test_iterative_structured_output_websearch_passes_tools(self, user_hierarchy_pydantic_model):
        """Iterative structured output can enable OpenAI Responses web search per field."""
        config = csv_to_llm.build_structured_output_config(
            model_reference=user_hierarchy_pydantic_model,
            model_class_name="User",
            output_field=None,
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
            iterate_fields=True,
            model_websearch=True,
        )
        mock_client = Mock()
        response = Mock()
        response.output_parsed = Mock(id=42)
        mock_client.responses.parse.return_value = response

        value = csv_to_llm._call_openai_structured_field(
            prompt_value="User record",
            structured_config=config,
            field_name="id",
            field_annotation=int,
            owner_models=[csv_to_llm._get_pydantic_model_class(user_hierarchy_pydantic_model, "User")],
            field_description=None,
            openai_client=mock_client,
        )

        assert value == 42
        assert mock_client.responses.parse.call_args.kwargs["tools"] == [{"type": "web_search"}]

    def test_iterative_field_model_name_respects_openai_limit(self):
        """Temporary per-field schema names must fit OpenAI's 64-character limit."""
        model_name = csv_to_llm._iterative_field_model_name(
            owner_names=[
                "EmailDeliveryService",
                "PricingAndProvisioningConfiguration",
                "DeeplyNestedAddonMetadata",
            ],
            field_name="requires_standalone_addon_subscription_before_provisioning",
        )

        assert len(model_name) <= 64
        assert model_name.startswith("CsvToLlm")

    def test_iterative_field_parallelism_uses_unused_parallel_budget(self):
        """Iterative field extraction should use spare parallel capacity without exceeding the row budget."""
        assert csv_to_llm._iterative_field_parallelism(parallel=64, total_tasks=1) == 64
        assert csv_to_llm._iterative_field_parallelism(parallel=64, total_tasks=2) == 32
        assert csv_to_llm._iterative_field_parallelism(parallel=64, total_tasks=64) == 1
        assert csv_to_llm._iterative_field_parallelism(parallel=64, total_tasks=750) == 1

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key'})
    def test_structured_output_invalid_field(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Invalid field names should raise before any API call is made."""
        with pytest.raises(ValueError, match="Field 'missing'"):
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                model="gpt-4o-mini",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_field="missing",
            )

    def test_perplexity_structured_iterative_rejected(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Perplexity structured output supports full-schema mode, not iterative OpenAI parse mode."""
        with pytest.raises(ValueError, match="pydantic-model-iterate"), \
             patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'key'}):
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                provider="perplexity",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_field="category",
                pydantic_model_iterate=True,
            )

    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key', 'OPENAI_API_KEY': ''})
    def test_perplexity_structured_output_flow(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Perplexity structured output should not require an OpenAI key."""
        with patch('csv_to_llm.core.call_perplexity_structured') as mock_structured:
            class Dummy(BaseModel):
                category: str = "Category"
                explanation: str = "Because"

            mock_structured.return_value = Dummy()

            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                provider="perplexity",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_field="category",
                test_first_row=True,
            )

            assert mock_structured.called
            df = pd.read_csv(output_csv_path)
            assert df.loc[0, 'response'] == "Category"

    def test_structured_output_field_prefix_conflict(self, sample_csv, output_csv_path, sample_pydantic_model):
        """Providing both field and prefix is rejected."""
        with pytest.raises(ValueError, match="cannot be used"), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'key'}):
            csv_to_llm.process_csv_with_claude(
                input_csv_path=sample_csv,
                output_csv_path=output_csv_path,
                prompt_template="Categorize: {description}",
                output_column="response",
                model="gpt-4o-mini",
                pydantic_model_path=sample_pydantic_model,
                pydantic_model_class="EmailCategory",
                pydantic_model_field="category",
                    pydantic_model_column_prefix="llm_",
                )

    def test_example_pydantic_models_are_loadable(self):
        """Bundled example schemas should be usable as structured-output models."""
        category_model = os.path.abspath("examples/email_category_model.py")
        broad_model = os.path.abspath("examples/email_classification_broad_model.py")

        category_config = csv_to_llm.build_structured_output_config(
            model_reference=category_model,
            model_class_name="EmailCategory",
            output_field="content_category",
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
        )
        broad_config = csv_to_llm.build_structured_output_config(
            model_reference=broad_model,
            model_class_name="EmailClassification",
            output_field="category",
            llm_model="gpt-5.2",
            max_tokens=1000,
            temperature=0,
            system_prompt="system",
        )

        assert category_config.output_field == "content_category"
        assert broad_config.output_field == "category"

    def test_pydantic_loader_rebuilds_common_typing_annotations(self, temp_dir):
        """Generated schemas with postponed Optional annotations should load."""
        model_path = os.path.join(temp_dir, "provider_headcount_model.py")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(
                "from __future__ import annotations\n"
                "from pydantic import BaseModel\n\n"
                "class ProviderHeadcountModel(BaseModel):\n"
                "    employee_count: Optional[int]\n"
            )

        model_cls = csv_to_llm._get_pydantic_model_class(
            model_path,
            "ProviderHeadcountModel",
        )

        assert model_cls.model_fields["employee_count"].annotation == Optional[int]

    def test_column_prefix_requires_pydantic_model(self, sample_csv, output_csv_path):
        """Providing a prefix without a Pydantic model should raise."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="--pydantic-model-column-prefix"):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Describe: {description}",
                    output_column="response",
                    pydantic_model_column_prefix="llm_",
                )


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

    def test_openai_responses_api_call(self):
        """OpenAI text generation should use the Responses API."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "OpenAI response"
        mock_client.responses.create.return_value = mock_response

        response = csv_to_llm.call_openai_api_uncached(
            client=mock_client,
            model="gpt-5.2",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="Test system",
            prompt_value="Test prompt",
        )

        assert response == "OpenAI response"
        mock_client.responses.create.assert_called_once()

    def test_openai_responses_api_call_with_web_search(self):
        """OpenAI text generation can enable Responses web search."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "OpenAI response"
        mock_client.responses.create.return_value = mock_response

        response = csv_to_llm.call_openai_api_uncached(
            client=mock_client,
            model="gpt-5.2",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="Test system",
            prompt_value="Test prompt",
            model_websearch=True,
        )

        assert response == "OpenAI response"
        assert mock_client.responses.create.call_args.kwargs["tools"] == [{"type": "web_search"}]

    def test_perplexity_chat_completion_call(self):
        """Perplexity text generation should use its OpenAI-compatible chat API."""
        mock_client = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Perplexity response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        response = csv_to_llm.call_perplexity_api_uncached(
            client=mock_client,
            model="sonar-pro",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="Test system",
            prompt_value="Test prompt",
        )

        assert response == "Perplexity response"
        mock_client.chat.completions.create.assert_called_once()

    def test_perplexity_responses_api_call_with_web_search(self):
        """Perplexity text generation can enable Responses web search tools."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Perplexity response"
        mock_client.responses.create.return_value = mock_response

        response = csv_to_llm.call_perplexity_api_uncached(
            client=mock_client,
            model="sonar-pro",
            max_tokens=1000,
            temperature=0.7,
            system_prompt="Test system",
            prompt_value="Test prompt",
            model_websearch=True,
        )

        assert response == "Perplexity response"
        kwargs = mock_client.responses.create.call_args.kwargs
        assert kwargs["model"] == "sonar-pro"
        assert kwargs["input"] == "Test prompt"
        assert kwargs["instructions"] == "Test system"
        assert kwargs["max_output_tokens"] == 1000
        assert kwargs["tools"] == [{"type": "web_search"}, {"type": "fetch_url"}]
        mock_client.chat.completions.create.assert_not_called()


class TestParallelProcessing(TestCsvToLlm):
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('concurrent.futures.process._check_system_limits', lambda: None)
    @patch('anthropic.Anthropic')
    def test_parallel_processing(self, mock_anthropic, sample_csv, output_csv_path, mock_anthropic_client):
        """Test parallel processing mode."""
        mock_anthropic.return_value = mock_anthropic_client
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
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
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
            mock_api.side_effect = Exception("API Error")
            with patch('csv_to_llm.core.call_claude_api_uncached', side_effect=Exception("API Error")):
                csv_to_llm.process_csv_with_claude(
                    input_csv_path=sample_csv,
                    output_csv_path=output_csv_path,
                    prompt_template="Describe: {description}",
                    output_column="response",
                    test_first_row=True
                )

        # Verify error was handled and saved to CSV
        df = pd.read_csv(output_csv_path)
        assert "LLM call failed" in str(df.loc[0, 'response'])
    
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
        
        with patch('csv_to_llm.core.call_claude_api_cached') as mock_api:
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


class TestAutoMode:

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_generates_files(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello", "World"], "body": ["a", "b"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="EmailCategory",
            python_code="from pydantic import BaseModel\n\nclass EmailCategory(BaseModel):\n    category: str\n",
            primary_field="category",
            prompt_template="Classify this subject: {subject}",
            output_column_name="llm_category",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="Categorize",
            input_csv_path=csv_path,
            sample_size=2,
            model="gpt-test",
            temperature=0,
            model_websearch=True,
            output_column=None,
            openai_client=None,
        )

        assert plan.prompt_template == design.prompt_template
        assert os.path.exists(plan.pydantic_model_path)
        assert plan.primary_field == "category"
        assert plan.output_column == "llm_category"
        parse_kwargs = mock_client.responses.parse.call_args.kwargs
        assert parse_kwargs["model"] == "gpt-test"
        assert parse_kwargs["tools"] == [{"type": "web_search"}]

    def test_run_auto_mode_writes_common_typing_imports(self, temp_dir):
        """Auto-generated model files should include common typing imports."""
        model_path = csv_to_llm_auto._ensure_python_file(
            "from __future__ import annotations\n"
            "from pydantic import BaseModel\n\n"
            "class ProviderHeadcountModel(BaseModel):\n"
            "    employee_count: Optional[int]\n",
            "ProviderHeadcountModel",
            temp_dir,
        )

        with open(model_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "from typing import Any, Dict, List, Optional, Union" in code
        assert "from pydantic import BaseModel, Field" in code
        assert code.startswith("from __future__ import annotations\nfrom typing")

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_openai_uses_cache(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello"], "body": ["a"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="EmailCategory",
            python_code="from pydantic import BaseModel\n\nclass EmailCategory(BaseModel):\n    category: str\n",
            primary_field="category",
            prompt_template="Classify this subject: {subject}",
            output_column_name="llm_category",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        first = run_auto_mode("Categorize cached", csv_path, 1, model="gpt-test")
        second = run_auto_mode("Categorize cached", csv_path, 1, model="gpt-test")

        assert first.prompt_template == second.prompt_template == design.prompt_template
        assert mock_client.responses.parse.call_count == 1

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_repairs_invalid_primary_field(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"provider_name": ["Shopify"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="ProviderHeadcountModel",
            python_code=(
                "from typing import Optional, Literal\n"
                "from pydantic import BaseModel, Field\n\n"
                "class ProviderHeadcountModel(BaseModel):\n"
                "    headcount_employee_count: Optional[int] = Field(default=None)\n"
                "    headcount_confidence: Literal['high', 'medium', 'low']\n"
                "    headcount_scope: Literal['global_company', 'unknown']\n"
                "    reasoning: str\n"
            ),
            primary_field="provider_headcount_answer",
            prompt_template="Estimate employee headcount for {provider_name}",
            output_column_name="provider_headcount_answer",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="How many employees does the company listed in provider_name have?",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
        )

        assert plan.primary_field == "headcount_employee_count"
        assert plan.output_column == "provider_headcount_answer"

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_invalid_primary_field_falls_back_to_multi_column(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="AmbiguousModel",
            python_code=(
                "from pydantic import BaseModel\n\n"
                "class AmbiguousModel(BaseModel):\n"
                "    category: str\n"
                "    action: str\n"
            ),
            primary_field="missing_answer",
            prompt_template="Classify {subject}",
            output_column_name="structured_payload",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="Classify this record",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
        )

        assert plan.primary_field is None
        assert plan.output_column == "structured_payload"

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_multi_column_updates_system_prompt(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello"], "body": ["a"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="EmailCategory",
            python_code="from pydantic import BaseModel\n\nclass EmailCategory(BaseModel):\n    category: str\n",
            primary_field=None,
            prompt_template="Classify this subject: {subject}",
            output_column_name="structured_payload",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        run_auto_mode(
            instruction="Categorize multi",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
            auto_multi_column=True,
        )

        system_prompt = mock_client.responses.parse.call_args.kwargs["input"][0]["content"]
        assert "one field per useful output column" in system_prompt
        assert "primary_field to null" in system_prompt

    def test_run_auto_mode_perplexity_generates_files(self, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello", "World"], "body": ["a", "b"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="EmailCategory",
            python_code="from pydantic import BaseModel\n\nclass EmailCategory(BaseModel):\n    category: str\n",
            primary_field="category",
            prompt_template="Classify this subject: {subject}",
            output_column_name="llm_category",
        )
        mock_response.output_text = design.model_dump_json()
        mock_client.responses.create.return_value = mock_response

        plan = run_auto_mode(
            instruction="Categorize",
            input_csv_path=csv_path,
            sample_size=2,
            provider="perplexity",
            model="pro-search",
            model_websearch=True,
            output_column=None,
            perplexity_client=mock_client,
        )

        assert plan.prompt_template == design.prompt_template
        assert os.path.exists(plan.pydantic_model_path)
        assert plan.primary_field == "category"
        assert plan.output_column == "llm_category"
        create_kwargs = mock_client.responses.create.call_args.kwargs
        assert create_kwargs["preset"] == "pro-search"
        assert create_kwargs["tools"] == [{"type": "web_search"}, {"type": "fetch_url"}]
        assert create_kwargs["response_format"]["type"] == "json_schema"
        assert create_kwargs["response_format"]["json_schema"]["name"] == "AutoModelDesign"

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test_perplexity_key"})
    @patch("csv_to_llm.auto.Perplexity")
    def test_run_auto_mode_perplexity_loads_api_key_from_env(self, mock_perplexity, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"subject": ["Hello"], "body": ["a"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_perplexity.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="EmailCategory",
            python_code="from pydantic import BaseModel\n\nclass EmailCategory(BaseModel):\n    category: str\n",
            primary_field="category",
            prompt_template="Classify this subject: {subject}",
            output_column_name="llm_category",
        )
        mock_response.output_text = design.model_dump_json()
        mock_client.responses.create.return_value = mock_response

        run_auto_mode(
            instruction="Categorize",
            input_csv_path=csv_path,
            sample_size=1,
            provider="perplexity",
            model="pro-search",
        )

        mock_perplexity.assert_called_once_with(api_key="test_perplexity_key")

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_escapes_literal_json_braces_in_prompt(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"Provider Name": ["Shopify"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="HostingClassification",
            python_code="from pydantic import BaseModel\n\nclass HostingClassification(BaseModel):\n    is_traditional_web_hosting: bool\n",
            primary_field="is_traditional_web_hosting",
            prompt_template=(
                "Classify {Provider Name}. Respond as JSON: {\n"
                '  "is_traditional_web_hosting": true or false,\n'
                '  "label": "Traditional Web Hosting" or "Not Traditional Web Hosting",\n'
                '  "reason_short": "very brief explanation"\n'
                "}"
            ),
            output_column_name="hosting_classification",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="Is this company a traditional web hosting provider?",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
            openai_client=None,
        )

        formatted = plan.prompt_template.format(**{"Provider Name": "Shopify"})
        assert "Classify Shopify" in formatted
        assert '"is_traditional_web_hosting": true or false' in formatted

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_escapes_nested_unknown_placeholder_in_json_example(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"Provider Name": ["Shopify"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="HostingClassification",
            python_code="from pydantic import BaseModel\n\nclass HostingClassification(BaseModel):\n    fit: str\n",
            primary_field="fit",
            prompt_template='Classify {Provider Name}. Return JSON: {"label": "{Fit}"}',
            output_column_name="hosting_classification",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="Is this company a traditional web hosting provider?",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
            openai_client=None,
        )

        formatted = plan.prompt_template.format(**{"Provider Name": "Shopify"})
        assert formatted == 'Classify Shopify. Return JSON: {"label": "{Fit}"}'

    def test_auto_prompt_escapes_remaining_unknown_fields(self):
        prompt = "Classify {Provider Name}. Use {Fit} as the label."
        escaped = _escape_unknown_prompt_fields(prompt, {"Provider Name"})

        assert escaped.format(**{"Provider Name": "Shopify"}) == "Classify Shopify. Use {Fit} as the label."

    @patch('csv_to_llm.auto.OpenAI')
    def test_run_auto_mode_supports_column_names_with_colons(self, mock_openai, temp_dir):
        csv_path = os.path.join(temp_dir, "auto.csv")
        pd.DataFrame({"Provider Name": ["Shopify"], "Fit: SMTP Relay": ["Poor Fit"]}).to_csv(csv_path, index=False)

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        design = AutoModelDesign(
            model_name="HostingClassification",
            python_code="from pydantic import BaseModel\n\nclass HostingClassification(BaseModel):\n    fit: str\n",
            primary_field="fit",
            prompt_template="Classify {Provider Name}. Existing SMTP relay fit: {Fit: SMTP Relay}",
            output_column_name="hosting_classification",
        )
        mock_response.output_parsed = design
        mock_client.responses.parse.return_value = mock_response

        plan = run_auto_mode(
            instruction="Is this company a traditional web hosting provider?",
            input_csv_path=csv_path,
            sample_size=1,
            model="gpt-test",
            openai_client=None,
        )

        rendered = csv_to_llm._render_prompt_template(
            plan.prompt_template,
            {"Provider Name": "Shopify", "Fit: SMTP Relay": "Poor Fit"},
            ["Provider Name", "Fit: SMTP Relay"],
        )
        assert rendered == "Classify Shopify. Existing SMTP relay fit: Poor Fit"


class TestCliAutoMode(TestCsvToLlm):

    def test_cli_auto_verbose_prints_prompt_and_passes_multi_column_flag(self, temp_dir, capsys):
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        model_path = os.path.join(temp_dir, "auto_model.py")
        pd.DataFrame({"subject": ["Hello"]}).to_csv(input_path, index=False)

        auto_plan = AutoPlan(
            prompt_template="Classify {subject}",
            pydantic_model_path=model_path,
            pydantic_model_class="AutoModel",
            primary_field=None,
            output_column="structured_payload",
        )

        argv = [
            "csv-to-llm",
            "--input", input_path,
            "--output", output_path,
            "--auto", "Classify multiple things",
            "--auto-multi-column",
            "--verbose",
        ]
        with patch.object(sys, "argv", argv), \
             patch("csv_to_llm.cli.run_auto_mode", return_value=auto_plan) as mock_auto, \
             patch("csv_to_llm.cli.process_csv_with_claude") as mock_process:
            csv_to_llm_cli.main()

        captured = capsys.readouterr()
        assert "Auto prompt:" in captured.out
        assert "Classify {subject}" in captured.out
        assert mock_auto.call_args.kwargs["auto_multi_column"] is True
        assert mock_process.call_args.kwargs["pydantic_model_column_prefix"] == "auto_"

    def test_cli_auto_multi_column_forces_flattening_even_with_primary_field(self, temp_dir):
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")
        model_path = os.path.join(temp_dir, "auto_model.py")
        pd.DataFrame({"subject": ["Hello"]}).to_csv(input_path, index=False)

        auto_plan = AutoPlan(
            prompt_template="Classify {subject}",
            pydantic_model_path=model_path,
            pydantic_model_class="AutoModel",
            primary_field="category",
            output_column="category",
        )

        argv = [
            "csv-to-llm",
            "--input", input_path,
            "--output", output_path,
            "--auto", "Classify multiple things",
            "--auto-multi-column",
        ]
        with patch.object(sys, "argv", argv), \
             patch("csv_to_llm.cli.run_auto_mode", return_value=auto_plan), \
             patch("csv_to_llm.cli.process_csv_with_claude") as mock_process:
            csv_to_llm_cli.main()

        process_kwargs = mock_process.call_args.kwargs
        assert process_kwargs["pydantic_model_field"] is None
        assert process_kwargs["pydantic_model_column_prefix"] == "auto_"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
